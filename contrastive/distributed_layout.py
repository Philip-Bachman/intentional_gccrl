"""Program definition for a distributed layout based on a builder."""

import dataclasses
import logging
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
from contrastive.environment_loop import FancyEnvironmentLoop
from contrastive.networks import apply_policy_and_sample


class DistributedLayout:
  """Program definition for a distributed agent based on a builder."""

  def __init__(
      self,
      seed,
      environment_factory,
      network_factory,
      builder,
      num_actors,
      actor_logger_fn = None,
      evaluator_factories = (),
      device_prefetch = True,
      prefetch_size = 1,
      max_number_of_steps = None,
      actor_observers = (),
      multithreading_colocate_learner_and_reverb = False,
      checkpointing_config = None,
      config = None):

    if prefetch_size < 0:
      raise ValueError(f'Prefetch size={prefetch_size} should be non negative')
    if actor_logger_fn is None:
      raise ValueError(f'actor_logger_fn={actor_logger_fn} should be a logger ')

    self._seed = seed
    self._builder = builder
    self._environment_factory = environment_factory
    self._network_factory = network_factory
    self._num_actors = num_actors
    self._device_prefetch = device_prefetch
    self._prefetch_size = prefetch_size
    self._max_number_of_steps = max_number_of_steps
    self._actor_logger_fn = actor_logger_fn
    self._evaluator_factories = evaluator_factories
    self._actor_observers = actor_observers
    self._multithreading_colocate_learner_and_reverb = (
        multithreading_colocate_learner_and_reverb)
    self._checkpointing_config = checkpointing_config
    self._config = config
    #
    # WARNING -- we assume all environments from self._environment_factory
    #            have the same spec!!!
    #
    dummy_seed = 1
    self._dummy_environment_spec = \
      specs.make_environment_spec(environment_factory(dummy_seed))
    print('********************************************')
    print('********************************************')
    print('self._dummy_environment_spec: {}'.format(self._dummy_environment_spec))
    print('********************************************')
    print('********************************************')

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
    rb_iterator = self._builder.make_dataset_iterator(
      replay, self._prefetch_size, self._device_prefetch
    )

    environment_key, actor_key = jax.random.split(random_key)

    # environments normally require uint32 as a seed.
    environment = self._environment_factory(
        utils.sample_uint32(environment_key))

    networks = self._network_factory(self._dummy_environment_spec)
    policy_fn = apply_policy_and_sample(networks)
    actor = self._builder.make_actor(actor_key, policy_fn, variable_source,
                                     rb_adder=rb_adder)

    # Create logger and counter.
    counter = counting.Counter(counter, 'actor')
    # Only actor #0 will write to bigtable in order not to spam it too much.
    logger = self._actor_logger_fn(actor_id)
    # Create the loop to connect environment and agent.
    return FancyEnvironmentLoop(environment, actor, counter,
                                logger, observers=self._actor_observers,
                                rb_iterator=rb_iterator,
                                rb_warmup=10)

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
      for evaluator in self._evaluator_factories:
        evaluator_key, key = jax.random.split(key)
        program.add_node(
            lp.CourierNode(evaluator, evaluator_key, learner, counter,
                          self._builder.make_actor))

    with program.group('actor'):
      for actor_id in range(self._num_actors):
        actor_key, key = jax.random.split(key)
        program.add_node(
            lp.CourierNode(self.actor, actor_key, replay, learner, counter,
                           actor_id))

    return program
