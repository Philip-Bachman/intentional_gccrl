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

"""A simple agent-environment training loop."""

import operator
import time
import collections
import tree
import inspect
from random import sample
from typing import List, Optional, Sequence

from acme import core
from acme import types
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.utils import signals

import dm_env
from dm_env import specs
import numpy as np
import jax


def has_named_argument(func, arg_name):
  """Check if 'func' has a parameter named 'arg_name'."""
  sig = inspect.signature(func)
  return arg_name in sig.parameters


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)


class SimpleReplayBuffer:
  def __init__(self, max_size):
    self.buffer = [None] * max_size
    self.max_size = max_size
    self.index = 0
    self.size = 0

  def __len__(self):
    return self.size

  def append(self, obj):
    self.buffer[self.index] = obj
    self.size = min(self.size + 1, self.max_size)
    self.index = (self.index + 1) % self.max_size

  def sample(self, batch_size):
    indices = sample(range(self.size), batch_size)
    return [self.buffer[index] for index in indices]


class FancyEnvironmentLoop(core.Worker):
  """A less simple RL environment loop that injects additional state.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. This can be used as:

    loop = FancyEnvironmentLoop(environment, actor)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.

  A list of 'Observer' instances can be specified to generate additional metrics
  to be logged by the logger. They have access to the 'Environment' instance,
  the current timestep datastruct and the current action.
  """

  def __init__(
      self,
      environment: dm_env.Environment,
      actor: core.Actor,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      update_actor_per: str = 'step',
      label: str = 'environment_loop',
      observers: Sequence[observers_lib.EnvLoopObserver] = (),
      use_env_goal: Optional[bool] = True,
      rb_warmup: Optional[int] = 100
  ):
    # check for some additional env features that we want...
    assert hasattr(environment, 'obs_dim')
    assert hasattr(environment, 'goal_dim')
    assert update_actor_per in ['step', 'episode']
    
    # Internalize agent and environment.
    self._environment = environment
    self._actor = actor
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        label, steps_key=self._counter.get_steps_key())
    self._update_actor_per = update_actor_per
    self._observers = observers

    # extra stuff for managing additional state
    self._use_env_goal = use_env_goal
    self._rb_warmup = rb_warmup
    self._obs_dim = environment.obs_dim
    self._goal_dim = environment.goal_dim
    self._goal_buffer = SimpleReplayBuffer(10000)

  def _get_goal(self, timestep):
    """Pull the goal from the given timestep's observation.
    """
    goal = timestep.observation[self._obs_dim:(self._obs_dim + self._goal_dim)].copy()
    return goal

  def _set_goal(self, timestep, new_goal=None):
    """Bypass the environment's goal and force a new goal.
    """
    if new_goal is not None:
      timestep.observation[self._obs_dim:(self._obs_dim + self._goal_dim)] = new_goal
    return timestep

  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    episode_start_time = time.time()
    episode_steps: int = 0
    episode_rnd_goal = None

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        self._environment.reward_spec())
    env_reset_start = time.time()
    timestep = self._environment.reset()

    # decide whether to sample a goal for this episode from the local buffer
    if (len(self._goal_buffer) > self._rb_warmup):
      if (not self._use_env_goal) and (np.random.rand() < 0.5):
        episode_rnd_goal = self._goal_buffer.sample(1)[0]
        timestep = self._set_goal(timestep, episode_rnd_goal)

    env_reset_duration = time.time() - env_reset_start

    if self._update_actor_per == 'episode':
      self._actor.update()

    # Make the first observation.
    self._actor.observe_first(timestep)
    for observer in self._observers:
      # Initialize the observer with the current state of the env after reset
      # and the initial timestep.
      observer.observe_first(self._environment, timestep)

    # Run an episode.
    while not timestep.last():
      # Give the actor the opportunity to update itself.
      if (self._update_actor_per == 'step'):
          self._actor.update()

      # Book-keeping.
      episode_steps += 1

      # Generate an action from the agent's policy.
      action = self._actor.select_action(timestep.observation)

      # Step the environment with the agent's selected action.
      timestep = self._environment.step(action)
      if (np.random.rand() < 0.01):
        # add a "viable" goal to the local goal buffer
        # -- for now, we treat visited states as viable goals
        self._goal_buffer.append(timestep.observation[:self._obs_dim])
      if episode_rnd_goal is not None:
        timestep = self._set_goal(timestep, episode_rnd_goal)

      # Have the agent and observers observe the timestep.
      self._actor.observe(action, next_timestep=timestep)
      for observer in self._observers:
        # One environment step was completed. Observe the current state of the
        # environment, the current timestep and the action.
        observer.observe(self._environment, timestep, action)

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_return = tree.map_structure(operator.iadd,
                                          episode_return,
                                          timestep.reward)

    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - episode_start_time)
    result = {
        'episode_length': episode_steps,
        'episode_return': episode_return,
        'steps_per_second': steps_per_second,
        'env_reset_duration_sec': env_reset_duration,
    }
    result.update(counts)
    for observer in self._observers:
      result.update(observer.get_metrics())
    return result

  def run(
      self,
      num_episodes: Optional[int] = None,
      num_steps: Optional[int] = None,
  ) -> int:
    """Perform the run loop.

    Run the environment loop either for `num_episodes` episodes or for at
    least `num_steps` steps (the last episode is always run until completion,
    so the total number of steps may be slightly more than `num_steps`).
    At least one of these two arguments has to be None.

    Upon termination of an episode a new episode will be started. If the number
    of episodes and the number of steps are not given then this will interact
    with the environment infinitely.

    Args:
      num_episodes: number of episodes to run the loop for.
      num_steps: minimal number of steps to run the loop for.

    Returns:
      Actual number of steps the loop executed.

    Raises:
      ValueError: If both 'num_episodes' and 'num_steps' are not None.
    """

    if not (num_episodes is None or num_steps is None):
      raise ValueError('Either "num_episodes" or "num_steps" should be None.')

    def should_terminate(episode_count: int, step_count: int) -> bool:
      return ((num_episodes is not None and episode_count >= num_episodes) or
              (num_steps is not None and step_count >= num_steps))

    episode_count: int = 0
    step_count: int = 0
    with signals.runtime_terminator():
      while not should_terminate(episode_count, step_count):
        episode_start = time.time()
        result = self.run_episode()
        result = {**result, **{'episode_duration': time.time() - episode_start}}
        episode_count += 1
        step_count += int(result['episode_length'])
        # Log the given episode results.
        self._logger.write(result)

    return step_count

