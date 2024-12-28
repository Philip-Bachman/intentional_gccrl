"""Program definition for a distributed layout based on a builder."""

import dataclasses
import logging
import functools
from typing import Any, Callable, Optional, Sequence

import dm_env
import jax
import launchpad as lp
import numpy as np
import reverb
import tqdm

from acme import core
from acme import specs
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import savers
from acme.jax import types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils

from default import make_default_logger
from contrastive.builder import ContrastiveBuilder
from contrastive.environment_loop import FancyEnvironmentLoop
from contrastive.networks import apply_policy_and_sample
from contrastive.utils import SuccessObserver, DistanceObserver


@dataclasses.dataclass
class CheckpointingConfig:  
  def __init__(
      self,
      save_dir = 'logs',
      add_uid = True):
    
    """Configuration options for learner checkpointer."""
    # The maximum number of checkpoints to keep.
    self.max_to_keep: int = 10
    # Which directory to put the checkpoint in.
    self.directory: str = save_dir
    # If True adds a UID to the checkpoint path, see
    # `paths.get_unique_id()` for how this UID is generated.
    self.add_uid: bool = add_uid


def get_actor_logger_fn(
    save_dir = "logs",
    add_uid = True,
    use_tboard = False):
  """Creates an actor logger."""

  def create_logger(actor_id):
    save_data = (actor_id == 0)
    return make_default_logger(
        'actor',
        save_data=save_data,
        save_dir=save_dir,
        add_uid=add_uid,
        steps_key='actor_steps',
        use_tboard=(use_tboard and save_data),
        use_term=save_data)
  return create_logger


def get_observers(config):
  observers = [
    SuccessObserver(),
    DistanceObserver(obs_dim=config.obs_dim,
                     goal_dim=config.goal_dim)
  ]
  return observers


class ContrastiveDistributedLayout:
  """Program definition for a distributed agent based on a builder."""

  def __init__(
      self,
      seed,
      config,
      environment_factory_train,
      environment_factory_eval,
      network_factory,
      num_actors,
      device_prefetch = True,
      max_number_of_steps = None,
      multithreading_colocate_learner_and_reverb = False):

    # Check that the environment-specific parts of the config have been set.
    assert config.max_episode_steps > 0
    assert config.obs_dim > 0
    if config.prefetch_size < 0:
      raise ValueError(f'Prefetch size={config.prefetch_size} should be non negative')

    save_dir = config.log_dir + config.alg_name + '_' + config.env_name + '_' + str(seed)

    self._seed = seed
    self._environment_factory_train = environment_factory_train
    self._environment_factory_eval = environment_factory_eval
    self._network_factory = network_factory
    self._num_actors = num_actors
    self._device_prefetch = device_prefetch
    self._prefetch_size = config.prefetch_size
    self._max_number_of_steps = max_number_of_steps
    self._actor_observers = get_observers(config)
    self._multithreading_colocate_learner_and_reverb = (
        multithreading_colocate_learner_and_reverb)
    self._config = config
    self._checkpointing_config = \
      CheckpointingConfig(save_dir, add_uid=config.add_uid)
    # WARNING -- we assume all environments from self._environment_factory_train
    #            have the same spec!!!
    dummy_seed = 1
    self._dummy_environment_spec = \
      specs.make_environment_spec(environment_factory_train(dummy_seed))
    
    # ...
    _learner_logger_fn = functools.partial(make_default_logger,
                                  'learner', save_data=True,
                                  asynchronous=True,
                                  serialize_fn=utils.fetch_devicearray,
                                  save_dir=save_dir,
                                  add_uid=config.add_uid,
                                  steps_key='learner_steps',
                                  use_term=False,
                                  use_tboard=True)
    self._builder = ContrastiveBuilder(config, logger_fn=_learner_logger_fn)

    # ...
    self._actor_logger_fn = get_actor_logger_fn(save_dir=save_dir,
                                                add_uid=config.add_uid,
                                                use_tboard=True)
    
    self._evaluator_logger_fn = functools.partial(
      make_default_logger,
      'evaluator', save_data=True,
      save_dir=save_dir,
      add_uid=config.add_uid,
      steps_key='actor_steps',
      use_term=False,
      use_tboard=True)

  def replay(self):
    """The replay storage."""
    replay_buffer = self._builder.make_replay_tables(self._dummy_environment_spec)
    return replay_buffer

  def counter(self):
    kwargs = {}
    if self._checkpointing_config:
      kwargs = vars(self._checkpointing_config)
    return savers.CheckpointingRunner(
        counting.Counter(),
        key='counter',
        subdirectory='counter',
        time_delta_minutes=self._config.time_delta_minutes,
        **kwargs)

  def learner(
      self,
      random_key,
      replay,
      counter,
  ):
    """The Learning part of the agent."""
    # create stuff that will be used by the learner
    networks = self._network_factory(self._dummy_environment_spec)
    iterator = self._builder.make_dataset_iterator(
      replay, self._prefetch_size, self._device_prefetch
    )
    counter = counting.Counter(counter, 'learner')

    learner = self._builder.make_learner(random_key, networks, iterator, counter)

    kwargs = {}
    if self._checkpointing_config:
      kwargs = vars(self._checkpointing_config)
    # Return the learning agent.
    return savers.CheckpointingRunner(
        learner,
        key='learner',
        subdirectory='learner',
        time_delta_minutes=self._config.time_delta_minutes,
        **kwargs)

  def actor(
      self,
      random_key,
      replay,
      variable_source,
      counter,
      actor_id
  ):
    """Actor process for interacting wth environment and collecting data."""
    rb_adder = self._builder.make_adder(replay)
    # create environment
    environment_key, actor_key = jax.random.split(random_key)
    environment = self._environment_factory_train(
        utils.sample_uint32(environment_key))
    # create networks
    networks = self._network_factory(self._dummy_environment_spec)
    policy_fn = apply_policy_and_sample(networks)
    actor = self._builder.make_actor(
      actor_key, policy_fn, variable_source, rb_adder=rb_adder
    )
    # create observers
    observers = get_observers(self._config)

    # Create logger and counter.
    counter = counting.Counter(counter, 'actor')
    # Only actor #0 will write to bigtable in order not to spam it too much.
    logger = self._actor_logger_fn(actor_id)
    # Create the loop to connect environment and agent.
    return FancyEnvironmentLoop(environment, actor, counter,
                                logger, observers=observers,
                                use_env_goal=self._config.use_env_goal,
                                update_actor_per='step',
                                rb_warmup=100)

  def evaluator(
      self,
      random_key,
      variable_source,
      counter
  ):
    """Evaluator process for interacting wth environment and collecting data."""
    # create environment
    environment_key, actor_key = jax.random.split(random_key)
    environment = self._environment_factory_eval(
      utils.sample_uint32(environment_key))
    # crate networks
    networks = self._network_factory(self._dummy_environment_spec)
    policy_fn = apply_policy_and_sample(networks, True)
    actor = self._builder.make_actor(
      actor_key, policy_fn, variable_source
    )
    # create observers
    observers = get_observers(self._config)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = self._evaluator_logger_fn()
    # Create the loop to connect environment and agent.
    return FancyEnvironmentLoop(environment, actor, counter,
                                logger, observers=observers,
                                update_actor_per='episode',
                                use_env_goal=True)

  def coordinator(
      self,
      counter,
      max_actor_steps
  ):
    steps_key = 'actor_steps'
    return lp_utils.StepsLimiter(counter, max_actor_steps, steps_key=steps_key)

  def build(
      self,
      name='agent',
      program=None
  ):
    """Build the distributed agent topology."""
    if not program:
      program = lp.Program(name=name)

    key = jax.random.PRNGKey(self._seed)

    replay_node = lp.ReverbNode(self.replay)
    with program.group('replay'):
      if self._multithreading_colocate_learner_and_reverb:
        replay = replay_node.create_handle()
      else:
        replay = program.add_node(replay_node)

    with program.group('counter'):
      counter = program.add_node(lp.CourierNode(self.counter))
      if self._max_number_of_steps is not None:
        _ = program.add_node(
            lp.CourierNode(self.coordinator, counter,
                           self._max_number_of_steps))

    learner_key, key = jax.random.split(key)
    learner_node = lp.CourierNode(self.learner, learner_key, replay, counter)
    with program.group('learner'):
      if self._multithreading_colocate_learner_and_reverb:
        learner = learner_node.create_handle()
        program.add_node(
            lp.MultiThreadingColocation([learner_node, replay_node]))
      else:
        learner = program.add_node(learner_node)

    with program.group('evaluator'):
      evaluator_key, key = jax.random.split(key)
      program.add_node(
          lp.CourierNode(self.evaluator, evaluator_key, learner, counter)
      )

    with program.group('actor'):
      for actor_id in range(self._num_actors):
        actor_key, key = jax.random.split(key)
        program.add_node(
            lp.CourierNode(self.actor, actor_key, replay, learner, counter,
                           actor_id))

    return program
