"""Contrastive RL networks definition."""
import dataclasses
from typing import Optional, Tuple, Callable

from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from itertools import product


# modified Tanh mean to be mapped to tanh(mean) to keep within [-1, 1]
from distributional import NormalTanhDistribution


@dataclasses.dataclass
class ContrastiveNetworks:
  """Network and pure functions for the Contrastive RL agent."""
  policy_network: networks_lib.FeedForwardNetwork
  q_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  repr_fn: Callable[Ellipsis, networks_lib.NetworkOutput]
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None


def apply_policy_and_sample(
    networks,
    eval_mode=False):
  """Returns a function that computes actions."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    return sample_fn(networks.policy_network.apply(params, obs), key)
  return apply_and_sample


def make_mlp(
    hidden_layer_sizes,
    out_size=None,
    out_layer=None,
    use_ln=True,
    cold_init=True
):
  assert (out_size is None) or (out_layer is None)         # this is just a simple function :-(
  assert not ((out_size is None) and (out_layer is None))  # this is just a simple function :-(
  # make mlp with given hidden layer sizes and output size,
  # with optional layernorm after each non-final linear layer.
  layer_sizes = hidden_layer_sizes
  is_final = [False for s in hidden_layer_sizes]
  if out_size is not None:
    layer_sizes = layer_sizes + (out_size,)
    is_final = is_final + [True]
  layer_list = []
  for lsz, isf in zip(layer_sizes, is_final):
    # add a linear layer, with possible "cold init" for final layer
    if isf and cold_init and (out_layer is None):
      layer_list.append(hk.Linear(lsz, w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')))
    else:
      layer_list.append(hk.Linear(lsz, w_init=hk.initializers.VarianceScaling(1e-1, 'fan_avg', 'uniform')))
    if not isf:
      # maybe add layernorm after all non-final linear layers
      if use_ln:
        layer_list.append(hk.LayerNorm(-1, True, True))
      # add relu after all non-final linear layers
      layer_list.append(jax.nn.relu)
  if out_layer is not None:
    layer_list.append(out_layer)
  mlp = hk.Sequential(layer_list)
  return mlp


def make_networks(
    spec,
    obs_dim,
    goal_dim,
    repr_dim = 64,
    hidden_layer_sizes = (256, 256),
    actor_min_std = 1e-2,
    use_image_obs = False):
  """Creates networks used by the agent."""

  num_dimensions = np.prod(spec.actions.shape, dtype=int)
  TORSO = networks_lib.AtariTorso

  def _unflatten_obs(obs_packed):
    state = jnp.reshape(obs_packed[:, :obs_dim], (-1, 64, 64, 3)) / 255.0
    intent = jnp.reshape(obs_packed[:, obs_dim:], (-1, 64, 64, 3)) / 255.0
    return state, intent

  def _repr_fn(obs_packed, action, goal):
    # obs_packed: should contain current state and intent policy conditions on
    # action: should contain action
    # goal: state we want to predict in the future
    #
    if use_image_obs:
      state, intent = _unflatten_obs(obs_packed)
      img_encoder = TORSO()
      state = img_encoder(state)
      intent = img_encoder(intent)
      goal = img_encoder(goal)
    else:
      state = obs_packed[:, :obs_dim]
      intent = obs_packed[:, obs_dim:]
    # TODO:  deal with conditioning on policy goal/intent
    intent = 0. * intent

    # encoder for (state, action, intent)
    sai_encoder = make_mlp(hidden_layer_sizes, out_size=repr_dim,
                           out_layer=None, use_ln=True, cold_init=True)
    sai_repr = sai_encoder(jnp.concatenate([state, action, intent], axis=-1))

    # encoder for goals (assume same format/type as state and intent)
    g_encoder = make_mlp(hidden_layer_sizes, out_size=repr_dim,
                         out_layer=None, use_ln=True, cold_init=True)
    g_repr = g_encoder(goal)
    return sai_repr, g_repr

  def _combine_repr(sai_repr, g_repr):
    return jax.numpy.einsum('ik,jk->ij', sai_repr, g_repr)

  def _critic_fn(obs_packed, action, goal):
    sai_repr, g_repr = _repr_fn(obs_packed, action, goal)
    critic_val = _combine_repr(sai_repr, g_repr)
    return critic_val, sai_repr, g_repr

  def _actor_fn(obs_packed):
    if use_image_obs:
      state, intent = _unflatten_obs(obs_packed)
      obs_packed = jnp.concatenate([state, intent], axis=-1)
      obs_packed = TORSO()(obs_packed)
    dist_layer = NormalTanhDistribution(num_dimensions, min_scale=actor_min_std, rescale=0.99)
    network = make_mlp(hidden_layer_sizes, out_size=None, out_layer=dist_layer, use_ln=True)
    return network(obs_packed)

  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn))
  repr_fn = hk.without_apply_rng(hk.transform(_repr_fn))

  # Create dummy observations and actions to create network parameters.
  # -- It's VERY important to note that the "observation" expected here is
  #    a "packed" observation that includes both a current environment state
  #    and a future goal state of the same form as the current state.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)       # obs is like [state; goal/intent]
  dummy_state = utils.zeros_like(dummy_obs[:obs_dim])
  dummy_goal = utils.zeros_like(dummy_obs[obs_dim:])  # intent and goal are same shape
  dummy_intent = utils.zeros_like(dummy_goal)  # intent and goal are same shape
  # ...  
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_state = utils.add_batch_dim(dummy_state)
  dummy_goal = utils.add_batch_dim(dummy_goal)
  dummy_intent = utils.add_batch_dim(dummy_intent)
  # packed observation, as fed to policy by the environment
  # -- make this a bit tedious, to belabour the point
  dummy_packed_obs = jnp.concatenate([dummy_state, dummy_intent], axis=-1)
  policy_network = networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, dummy_packed_obs), policy.apply)
  q_network = networks_lib.FeedForwardNetwork(
          lambda key: critic.init(key, dummy_packed_obs, dummy_action, dummy_goal), critic.apply)

  return ContrastiveNetworks(
      policy_network=policy_network,
      q_network=q_network,
      repr_fn=repr_fn.apply,
      log_prob=lambda params, actions: params.log_prob(actions),
      sample=lambda params, key: params.sample(seed=key),
      sample_eval=lambda params, key: params.mode(),
      )
