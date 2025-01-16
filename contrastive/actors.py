# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple JAX actors."""

from typing import Generic, Optional

from acme import adders
from acme import core
from acme import types
from acme.agents.jax import actor_core
from acme.jax import networks as network_lib
from acme.jax import utils
from acme.jax import variable_utils
import dm_env
import jax


class GenericActor(core.Actor, Generic[actor_core.State, actor_core.Extras]):
  """A generic actor implemented on top of ActorCore.

  An actor based on a policy which takes observations and outputs actions. It
  also adds experiences to replay and updates the actor weights from the policy
  on the learner.
  """

  def __init__(
      self,
      actor: actor_core.ActorCore[actor_core.State, actor_core.Extras],
      random_key: network_lib.PRNGKey,
      variable_client: Optional[variable_utils.VariableClient],
      adder: Optional[adders.Adder] = None,
      jit: bool = True,
      backend: Optional[str] = 'cpu'
  ):
    """Initializes a feed forward actor.

    Args:
      actor: actor core.
      random_key: Random key.
      variable_client: The variable client to get policy parameters from.
      adder: An adder to add experiences to.
      jit: Whether or not to jit the passed ActorCore's pure functions.
      backend: Which backend to use when jitting the policy.
    """
    self._random_key = random_key
    self._variable_client = variable_client
    self._adder = adder
    self._state = None

    # Unpack ActorCore, jitting if requested.
    if jit:
      self._init = jax.jit(actor.init, backend=backend)
      self._policy = jax.jit(actor.select_action, backend=backend)
    else:
      self._init = actor.init
      self._policy = actor.select_action
    self._get_extras = actor.get_extras

  @property
  def _params(self):
    return self._variable_client.params if self._variable_client else []

  def select_action(self, observation: network_lib.Observation) -> types.NestedArray:
    action, self._state = self._policy(self._params, observation, self._state)
    return utils.to_numpy(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._random_key, key = jax.random.split(self._random_key)
    self._state = self._init(key)
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: network_lib.Action, next_timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add(action, next_timestep,
                      extras=self._get_extras(self._state))

  def update(self, wait: bool = False):
    if self._variable_client:
      self._variable_client.update(wait)


class ContrastiveGaussianActor(GenericActor):
  """Actor that adds the option to start training with random policy.
  """
  def __init__(
      self,
      actor: actor_core.ActorCore[actor_core.State, actor_core.Extras],
      random_key: network_lib.PRNGKey,
      variable_client: Optional[variable_utils.VariableClient],
      adder: Optional[adders.Adder] = None,
      jit: bool = True,
      backend: Optional[str] = 'cpu',
      initially_random: bool = False
  ):
    assert (variable_client is not None)
    self._initially_random = initially_random
    super().__init__(
        actor=actor,
        random_key=random_key,
        variable_client=variable_client,
        adder=adder,
        jit=jit,
        backend=backend
    )

  def select_action(self,
                    observation):
    # decide whether agent has been updated based on whether some params
    # in the "distributional" layer of policy are still precisely 0
    # -- checking equality on floats is a questionable practice, lol
    # -- assuming 0 init for params is kinda iffy, lol
    if self._initially_random and (self._params['Distribution/~/linear']['b'] == 0).all():
      shape = self._params['Distribution/~/linear']['b'].shape
      rng, self._state = jax.random.split(self._state)
      action = jax.random.uniform(key=rng, shape=shape,
                                  minval=-1.0, maxval=1.0)
    else:
      action, self._state = self._policy(self._params, observation,
                                         self._state)
    return utils.to_numpy(action)

