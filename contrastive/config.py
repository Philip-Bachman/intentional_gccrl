"""Contrastive RL config."""
import dataclasses
from typing import Any, Optional, Union, Tuple

from acme import specs
from acme.adders import reverb as adders_reverb
import numpy as onp


@dataclasses.dataclass
class ContrastiveConfig:
  """Configuration options for contrastive RL."""
  add_uid: bool = True
  time_delta_minutes: int = 100000
  log_dir: str = 'logs/'
  env_name: str = ''
  alg_name: str = ''
  seed: int = 0
  max_number_of_steps: int = 10_000_000
  num_actors: int = 4

  # env options
  # use_env_goal : whether to use the fixed env goal during training rollouts
  #                note - the fixed env goal is always used in evaluators
  use_env_goal: bool = True
    
  # Loss options
  batch_size: int = 384
  learning_rate: float = 3e-4
  reward_scale: float = 1
  discount: float = 0.99
  n_step: int = 1
  # Target smoothing coefficient.
  tau: float = 0.005
  hidden_layer_sizes: Tuple[int, Ellipsis] = (1024, 1024, 1024, 1024)
  
  # Loss options - entropy
  # Coefficient applied to the entropy bonus. If None, an adaptative
  # coefficient will be used.
  use_action_entropy: bool = False
  entropy_alpha: float = 0.0
  actor_min_std: float = 1e-3

  # Replay options
  min_replay_size: int = 10000
  max_replay_size: int = 1000000
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  prefetch_size: int = 4
  num_parallel_calls: Optional[int] = 4
  samples_per_insert: float = 256
  # Rate to be used for the SampleToInsertRatio rate limitter tolerance.
  # See a formula in make_replay_buffer for more details.
  samples_per_insert_tolerance_rate: float = 0.1
  num_sgd_steps_per_step: int = 64  # Gradient updates to perform per step.
  
  # training options
  no_repr: bool = False
  repr_dim: Union[int, str] = 64  # size of infonce representation
  use_random_actor: bool = True   # warmup with uniform random policy
  use_cpc: bool = False
  use_td: bool = False
  use_image_obs: bool = False
  random_goals: int = 1  # switch for how to sample actor training goals
  jit: bool = True
  add_mc_to_td: bool = False
  resample_neg_actions: bool = False
  
  # Parameters that should be overwritten, based on each environment.
  obs_dim: int = -1
  goal_dim: int = -1
  latent_dim: int = -1
  max_episode_steps: int = -1


