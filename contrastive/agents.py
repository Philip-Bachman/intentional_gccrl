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

    super().__init__(
        seed=seed,
        config=config,
        environment_factory_train=environment_factory_train,
        environment_factory_eval=environment_factory_eval,
        network_factory=network_factory,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=config.prefetch_size
    )
