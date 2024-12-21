"""Defines distributed contrastive RL agents, using JAX."""

import functools
import dataclasses
from typing import Callable, Optional, Sequence

import jax
from acme import specs
from acme.jax import utils
from acme.utils import counting

import contrastive.builder as builder
import contrastive.config as contrastive_config
import contrastive.distributed_layout as distributed_layout
import contrastive.networks as networks
import contrastive.utils as contrastive_utils
from contrastive.environment_loop import FancyEnvironmentLoop

from default import make_default_logger


def default_evaluator_factory(
    config,
    environment_factory,
    network_factory,
    save_dir = "logs",
    add_uid = True):
  """Returns a default evaluator process."""
  def evaluator(
      random_key,
      variable_source,
      counter,
      make_actor,
  ):
    """The evaluation process."""

    # Create environment and evaluator networks
    environment_key, actor_key = jax.random.split(random_key)
    # Environments normally require uint32 as a seed.
    environment = environment_factory(utils.sample_uint32(environment_key))
    nets = network_factory(specs.make_environment_spec(environment))

    actor = make_actor(
      actor_key,
      networks.apply_policy_and_sample(nets, True),
      variable_source
    )

    observers = [
        contrastive_utils.SuccessObserver(),
        contrastive_utils.DistanceObserver(
            obs_dim=config.obs_dim,
            goal_dim=config.goal_dim)
    ]

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = make_default_logger('evaluator', save_data=True, save_dir=save_dir,
                                 add_uid=add_uid, steps_key='actor_steps',
                                 use_term=False, use_tboard=True)

    # Create the run loop and return it.
    return FancyEnvironmentLoop(environment, actor, counter,
                                logger, observers=observers,
                                update_actor_per='episode',
                                use_env_goal=True)
  return evaluator


class ContrastiveDistributedLayout(distributed_layout.DistributedLayout):
  """Distributed program definition for contrastive RL."""

  def __init__(
      self,
      environment_factory_train,
      environment_factory_eval,
      network_factory,
      config,
      seed,
      num_actors,
      max_number_of_steps = None,
  ):
    # Check that the environment-specific parts of the config have been set.
    assert config.max_episode_steps > 0
    assert config.obs_dim > 0

    save_dir = config.log_dir + config.alg_name + '_' + config.env_name + '_' + str(seed)

    # ...
    evaluator_factories = [
        default_evaluator_factory(
            config=config,
            environment_factory=environment_factory_eval,
            network_factory=network_factory,
            save_dir=save_dir,
            add_uid=config.add_uid)
    ]

    super().__init__(
        seed=seed,
        config=config,
        environment_factory_train=environment_factory_train,
        environment_factory_eval=environment_factory_eval,
        network_factory=network_factory,
        evaluator_factories=evaluator_factories,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=config.prefetch_size
    )
