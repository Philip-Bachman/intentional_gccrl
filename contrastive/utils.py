"""Utilities for the contrastive RL agent."""
import functools
from typing import Dict
from typing import Optional, Sequence

from acme import types
from acme.agents.jax import actors
from acme.jax import networks as network_lib
from acme.jax import utils
from acme.utils.observers import base as observers_base
from acme.wrappers import base
from acme.wrappers import canonical_spec
from acme.wrappers import gym_wrapper
from acme.wrappers import step_limit
import dm_env
import env_utils
import jax
import numpy as np
import os


class SuccessObserver(observers_base.EnvLoopObserver):
  """Observe each episode and report success if any rewards are > 0.

  This success metric assumes (roughly) that we're watching the agent act in
  a goal reaching environmment with indicator reward for reaching the goal.
  """
  def __init__(self):
    self._rewards = []
    self._success = []

  def observe_first(self, env, timestep):
    """What to do following environment reset."""
    # If self._rewards is not empty list, then the environment was reset after
    # a completed episode observed by this observer.
    if self._rewards:
      success = np.sum(self._rewards) >= 1
      self._success.append(success)
    # set list of observed rewards back to empty to start this new episode
    self._rewards = []

  def observe(self, env, timestep, action):
    """Record reward from one environment step."""
    assert timestep.reward in [0, 1]  # enforce "goal reaching" task assumption
    self._rewards.append(timestep.reward)

  def get_metrics(self):
    """Returns metrics collected for the current episode."""
    if len(self._rewards) > 0:
      success_1 = float(np.sum(self._rewards) >= 1)
    else:
      success_1 = 0.
    if len(self._success) < 1000:
      if len(self._success) > 1:
        success_1k = np.mean(self._success)
      else:
        success_1k = 0.
    else:
      success_1k = np.mean(self._success[-1000:])
    return {
        'success': success_1,
        'success_1000': success_1k,
    }


class DistanceObserver(observers_base.EnvLoopObserver):
  """Observer that measures the L2 distance to the goal."""

  def __init__(self, obs_dim, goal_dim,
               smooth = True):
    self._distances = []
    self._obs_dim = obs_dim
    self._goal_dim = goal_dim
    self._smooth = smooth
    self._history = {}

  def _get_distance(self, env, timestep):
    if hasattr(env, '_dist'):
      assert env._dist
      return env._dist[-1]
    else:
      # if environment doesn't provide a built-in distance metric, then
      # we'll just use simple euclidean distance.
      # -- we assume packed observation like [state; policy goal]
      obs = timestep.observation[:self._obs_dim]
      goal = timestep.observation[self._obs_dim:(self._obs_dim + self._goal_dim)]
      dist = np.linalg.norm(obs - goal)
      return dist

  def observe_first(self, env, timestep):
    """Observes the initial state."""
    if self._smooth and self._distances:
      for key, value in self._get_current_metrics().items():
        self._history[key] = self._history.get(key, []) + [value]
    self._distances = [self._get_distance(env, timestep)]

  def observe(self, env, timestep, action):
    """Records one environment step."""
    self._distances.append(self._get_distance(env, timestep))

  def _get_current_metrics(self):
    metrics = {
        'init_dist': self._distances[0],
        'final_dist': self._distances[-1],
        'delta_dist': self._distances[0] - self._distances[-1],
        'min_dist': min(self._distances),
    }
    return metrics

  def get_metrics(self):
    """Returns metrics collected for the current episode."""
    metrics = self._get_current_metrics()
    if self._smooth:
      for key, vec in self._history.items():
        for size in [10, 100, 1000]:
          metrics['%s_%d' % (key, size)] = np.nanmean(vec[-size:])
    return metrics


def make_environment(env_name, seed,
                     latent_dim=None,
                     fixed_goal=None):
  """Creates the environment.

  Args:
    env_name: name of the environment
    seed: seed for loading environment
    latent_dim: latent dim for environment
    fixed_goal: fixed goal location?
    return_extra: whether to return some info about the environment
                  in addition to the environment

    seed: random seed.
  Returns:
    env: the environment
  """

  np.random.seed(seed)
  gym_env, obs_dim, max_episode_steps = env_utils.load(env_name, fixed_goal)
  env = gym_wrapper.GymWrapper(gym_env)
  env = step_limit.StepLimitWrapper(env, step_limit=max_episode_steps)
  return env

