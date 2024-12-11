"""Defines distributed contrastive RL agents, using JAX."""

import functools
import dataclasses
from typing import Callable, Optional, Sequence

import jax
from acme import specs
from acme.jax import utils
from acme.utils import counting

from contrastive import builder
from contrastive import config as contrastive_config
from contrastive import distributed_layout
from contrastive import networks
from contrastive import utils as contrastive_utils
from contrastive.environment_loop import FancyEnvironmentLoop

from default import make_default_logger


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
    log_every = 10,
    save_dir = "logs",
    add_uid = True,
    use_tboard = False):
  """Creates an actor logger."""

  if use_tboard:
    keys_tboard = ['']

  def create_logger(actor_id):
    return make_default_logger(
        'actor',
        save_data=(actor_id == 0),
        save_dir=save_dir,
        add_uid=add_uid,
        time_delta=log_every,
        steps_key='actor_steps',
        use_tboard=use_tboard,
        use_term=True)
  return create_logger


def default_evaluator_factory(
    environment_factory,
    network_factory,
    policy_factory,
    observers = (),
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
    networks = network_factory(specs.make_environment_spec(environment))

    actor = make_actor(actor_key, policy_factory(networks), variable_source)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = make_default_logger('evaluator', save_data=True, save_dir=save_dir,
                                 add_uid=add_uid, steps_key='actor_steps',
                                 use_term=False)

    # Create the run loop and return it.
    return FancyEnvironmentLoop(environment, actor, counter,
                                logger, observers=observers)
  return evaluator



class ContrastiveDistributedLayout(distributed_layout.DistributedLayout):
  """Distributed program definition for contrastive RL."""

  def __init__(
      self,
      environment_factory,
      environment_factory_fixed_goals,
      network_factory,
      config,
      seed,
      num_actors,
      max_number_of_steps = None,
      log_every = 10.0
  ):
    # Check that the environment-specific parts of the config have been set.
    assert config.max_episode_steps > 0
    assert config.obs_dim > 0

    logger_fn = functools.partial(make_default_logger,
                                  'learner', save_data=True,
                                  time_delta=log_every, asynchronous=True,
                                  serialize_fn=utils.fetch_devicearray,
                                  save_dir = config.log_dir + config.alg_name + '_' 
                                  + config.env_name + '_' + str(seed),
                                  add_uid = config.add_uid,
                                  steps_key='learner_steps', use_term=False,
                                  use_tboard=True)
    contrastive_builder = builder.ContrastiveBuilder(config, logger_fn=logger_fn)
    
    # ...
    eval_policy_factory = (
        lambda n: networks.apply_policy_and_sample(n, True))
    eval_observers = [
        contrastive_utils.SuccessObserver(),
        contrastive_utils.DistanceObserver(
            obs_dim=config.obs_dim,
            start_index=config.start_index,
            end_index=config.end_index)
    ]
    evaluator_factories = [
        default_evaluator_factory(
            environment_factory=environment_factory_fixed_goals,
            network_factory=network_factory,
            policy_factory=eval_policy_factory,
            observers=eval_observers,
            save_dir = config.log_dir + config.alg_name + '_'
            + config.env_name + '_' + str(seed),
            add_uid = config.add_uid)
    ]
    if config.local:
        evaluator_factories = []

    # ...
    actor_observers = [
        contrastive_utils.SuccessObserver(),
        contrastive_utils.DistanceObserver(obs_dim=config.obs_dim,
                                           start_index=config.start_index,
                                           end_index=config.end_index)]
    actor_logger_dir = config.log_dir + config.alg_name + '_' + config.env_name + '_' + str(seed)
    actor_logger_fn = get_actor_logger_fn(log_every,
                                          save_dir=actor_logger_dir,
                                          add_uid=config.add_uid,
                                          use_tboard=True)
    super().__init__(
        seed=seed,
        environment_factory=environment_factory,
        environment_factory_fixed_goals=environment_factory_fixed_goals,
        network_factory=network_factory,
        builder=contrastive_builder,
        policy_network=networks.apply_policy_and_sample,
        evaluator_factories=evaluator_factories,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=config.prefetch_size,
        actor_logger_fn=actor_logger_fn,
        observers=actor_observers,
        checkpointing_config=CheckpointingConfig(
            save_dir = config.log_dir + config.alg_name + '_'
            + config.env_name + '_' + str(seed), add_uid=config.add_uid),
        config=config)
