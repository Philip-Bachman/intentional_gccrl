
r"""Example running contrastive RL in JAX.

Run using multi-threading
  python lp_contrastive.py --lp_launch_type=local_mt


"""
import functools
import logging
from typing import Any, Dict

from absl import app
from absl import flags

import launchpad as lp
import numpy as np
import os

import contrastive


FLAGS = flags.FLAGS

flags.DEFINE_string('log_dir_path', 'logs/', 'Where to log metrics')
flags.DEFINE_integer('time_delta_minutes', 5, 'how often to save checkpoints')
flags.DEFINE_integer('seed', 42, 'Specify seed, only used if use_slurm_array is false')
flags.DEFINE_bool('add_uid', True, 'Whether to add a unique id to the log directory name')
flags.DEFINE_string('alg', 'contrastive_cpc', 'Algorithm type, e.g. default is contrastive_cpc with no entropy or KL losses')
flags.DEFINE_string('env', 'sawyer_bin', 'Environment type, e.g. default is sawyer bin')
flags.DEFINE_integer('num_steps', 8_000_000, 'Number of steps to run', lower_bound=0)
flags.DEFINE_bool('sample_goals', False, 'sample the goal position uniformly according to the environment (corresponds to the original contrastive_rl algorithm)')

# fixed goal coordinates for supported environments
fixed_goal_dict={'point_Spiral11x11': [np.array([5,5], dtype=float), np.array([10,10], dtype=float)],
                     #note: sawyer fixed goal positions vary slightly with each episode
                      'sawyer_bin': np.array([0.12, 0.7, 0.02]),
                      'sawyer_box': np.array([0.0, 0.75, 0.133]),
                      'sawyer_peg': np.array([-0.3, 0.6, 0.0])}


def get_program(params):
  """Constructs the program."""

  env_name = params['env_name']
  seed = params['seed']

  config = contrastive.ContrastiveConfig(**params)
  
  fix_goals = params['fix_goals']

  if fix_goals:
    fixed_goal = fixed_goal_dict[env_name]
  else:
    fixed_goal = None

  environment, obs_dim, goal_dim = \
    contrastive.make_environment(env_name, seed=seed, latent_dim=None,
                                 fixed_goal=fixed_goal, return_extra=True)

  assert (environment.action_spec().minimum == -1).all()
  assert (environment.action_spec().maximum == 1).all()
  config.obs_dim = obs_dim
  config.goal_dim = goal_dim
  config.max_episode_steps = getattr(environment, '_step_limit') + 1
  network_factory = functools.partial(
      contrastive.make_networks,
      obs_dim=config.obs_dim,
      goal_dim=config.goal_dim,
      repr_dim=config.repr_dim,
      use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes)
  
  # factory for training environments (may sample goals)
  env_factory = lambda seed: contrastive.make_environment(
      env_name, seed, latent_dim=None, fixed_goal=fixed_goal,
      return_extra=False)
  # factory for evaluation environments (use fixed goals)
  env_factory_fixed_goals = lambda seed: contrastive.make_environment(
      env_name, seed, latent_dim=None, fixed_goal=fixed_goal_dict[env_name],
      return_extra=False)
    
  agent = contrastive.ContrastiveDistributedLayout(
      seed=seed,
      environment_factory=env_factory,
      environment_factory_fixed_goals=env_factory_fixed_goals,
      network_factory=network_factory,
      config=config,
      num_actors=config.num_actors,
      max_number_of_steps=config.max_number_of_steps)
  return agent.build()


def main(_):
  # Create experiment description.

  # 1. Select an environment.
  # Supported environments:
  #   Metaworld: sawyer_{bin,box,peg}
  #   2D nav: point_{Spiral11x11}
  env_name = FLAGS.env
  print('Using env {}...'.format(env_name))
  
  seed_idx = FLAGS.seed
  print('Using random seed {}...'.format(seed_idx))
  params = {
      'seed': seed_idx,
      'use_random_actor': True,
      'entropy_coefficient': 0.0,
      'env_name': env_name,
      # the number of environment steps
      'max_number_of_steps': FLAGS.num_steps,
  }
  # 2. Select an algorithm. The currently-supported algorithms are:
  # contrastive_nce, contrastive_cpc, c_learning, nce+c_learning
  # Many other algorithms can be implemented by passing other parameters
  # or adding a few lines of code.
  # By default, do contrastive CPC
  alg = FLAGS.alg
  print('Using alg {}...'.format(alg))
  params['alg_name'] = alg
  params['fix_goals'] = not FLAGS.sample_goals
  add_uid = FLAGS.add_uid
  params['add_uid'] = add_uid
  print('Adding uid: {}...'.format(params['add_uid']))
  
  params['log_dir'] = FLAGS.log_dir_path
  params['time_delta_minutes'] = FLAGS.time_delta_minutes
  
  if alg == 'contrastive_cpc':
    params['use_cpc'] = True
  elif alg == 'c_learning':
    assert False
  elif alg == 'nce+c_learning':
    assert False
  else:
    raise NotImplementedError('Unknown method: %s' % alg)

  program = get_program(params)
  # Set terminal='tmux' if you want different components in different windows.
  
  print(params)

  # squash some annoying TFlow deprecation warnings
  logger = logging.getLogger()
  class CheckTypesFilter(logging.Filter):
      def filter(self, record):
          return "check_types" not in record.getMessage()
  logger.addFilter(CheckTypesFilter())
  
  lp.launch(program, terminal='current_terminal')

if __name__ == '__main__':
  app.run(main)
