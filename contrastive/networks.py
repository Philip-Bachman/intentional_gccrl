"""Contrastive RL networks definition."""
import dataclasses
from typing import Optional, Tuple, Callable

from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.networks.base import NetworkOutput, Action, LogProb, Value, \
                                   PRNGKey, FeedForwardNetwork
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from itertools import product


# modified Tanh mean to be mapped to tanh(mean) to keep within [-1, 1]
from contrastive.distributional import NormalTanhDistribution


# recapitulate some typedefs from acme.jax.networks...
LogProbFn = Callable[[NetworkOutput, Action], LogProb]
SampleFn  = Callable[[NetworkOutput, PRNGKey], Action]


@dataclasses.dataclass
class ContrastiveNetworks:
  """Network and pure functions for the Contrastive RL agent."""
  policy_network: FeedForwardNetwork
  q_network: FeedForwardNetwork
  log_prob: LogProbFn
  sample: SampleFn
  sample_eval: Optional[SampleFn] = None


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
  for l_sz, is_f in zip(layer_sizes, is_final):
    if is_f and cold_init and (out_layer is None):
      # this should be for final layers in the critic
      layer_list.append(hk.Linear(l_sz, w_init=hk.initializers.VarianceScaling(1e-1, 'fan_avg', 'uniform')))
    else:
      # this should be for hidden layers in the actor and critic
      layer_list.append(hk.Linear(l_sz, w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')))
    # add normalization and non-linearity on hidden layers
    if not is_f:
      if use_ln:
        layer_list.append(hk.LayerNorm(-1, True, True))
      layer_list.append(jax.nn.relu)
  # add an extra output layer for the actor (probably tfd distribution)
  if out_layer is not None:
    layer_list.append(out_layer)
  mlp = hk.Sequential(layer_list)
  return mlp


def make_networks(
    spec,
    obs_dim,
    repr_dim = 64,
    hidden_layer_sizes = (256, 256),
    actor_min_std = 1e-2,
    use_image_obs = False,
    use_policy_goal_critic = True,
    use_policy_goal_actor = True):
  """Creates networks used by the agent."""
  assert (not use_image_obs)  # TODO: patch things up to handle image observations
  action_dim = np.prod(spec.actions.shape, dtype=int)
  TORSO = networks_lib.AtariTorso

  def _unflatten_obs(obs, is_packed=True):
    state = jnp.reshape(obs[:, :obs_dim], (-1, 64, 64, 3)) / 255.0
    if is_packed:
      goal = jnp.reshape(obs[:, obs_dim:], (-1, 64, 64, 3)) / 255.0
    else:
      goal = None
    return state, goal

  def _repr_fn(obs_packed, action, pert_goal):
    # obs_packed : should contain current state and policy goal
    # action     : should contain action
    # pert_goal  : state we want to predict in the future
    state = obs_packed[:, :obs_dim]
    policy_goal = obs_packed[:, obs_dim:]
    if not use_policy_goal_critic:
      policy_goal = 0. * policy_goal

    # encoder for (state, action, policy goal)
    sag_encoder = make_mlp(hidden_layer_sizes, out_size=repr_dim,
                           out_layer=None, use_ln=True, cold_init=True)
    sag_repr = sag_encoder(jnp.concatenate([state, action, policy_goal], axis=-1))

    # encoder for perturbation goals
    g_encoder = make_mlp(hidden_layer_sizes, out_size=repr_dim,
                         out_layer=None, use_ln=True, cold_init=True)
    g_repr = g_encoder(pert_goal)
    return sag_repr, g_repr

  def _combine_repr(sag_repr, g_repr):
    return jax.numpy.einsum('ik,jk->ij', sag_repr, g_repr)

  def _critic_fn(obs_packed, action, pert_goal):
    sag_repr, g_repr = _repr_fn(obs_packed, action, pert_goal)
    critic_val = _combine_repr(sag_repr, g_repr)
    return critic_val, sag_repr, g_repr

  def _actor_fn(obs_packed):
    # input like [state; policy goal] or [state; policy goal; perturbation goal]
    op_dim = obs_packed.shape[-1]
    state = obs_packed[:, :obs_dim]
    policy_goal = obs_packed[:, obs_dim:(2 * obs_dim)]
    if op_dim == (2 * obs_dim):
      # when observation doesn't include an explicit perturbation goal, we
      # assume the perturbation and policy goals are the same
      # -- this should mostly happen when acting in the environment
      pert_goal = policy_goal
    elif op_dim == (3 * obs_dim):
      # packed observation includes a policy goal and a perturbation goal...
      pert_goal = obs_packed[:, (2 * obs_dim):]
    else:
      # packed observation was not shaped properly...
      assert False

    if use_policy_goal_actor:
      obs_packed = jnp.concatenate([state, policy_goal, pert_goal], axis=-1)
    else:
      obs_packed = jnp.concatenate([state, pert_goal, pert_goal], axis=-1)

    # full packed input to actor like:
    # -- [state; policy goal; perturbation goal]
    dist_layer = NormalTanhDistribution(
      action_dim, min_scale=actor_min_std, rescale=0.99)
    network = make_mlp(
      hidden_layer_sizes, out_size=None, out_layer=dist_layer, use_ln=True)
    return network(obs_packed)

  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn))

  # create dummy observations and actions to create network parameters.
  # -- it's important to note that the "observation" expected here is a
  #    "packed" observation that includes both a current environment state
  #    and a future goal state of the same form as the current state.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)       # obs is like [state; goal]
  dummy_state = utils.zeros_like(dummy_obs[:obs_dim])
  dummy_goal = utils.zeros_like(dummy_obs[obs_dim:])
  # ...  
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_state = utils.add_batch_dim(dummy_state)
  dummy_goal = utils.add_batch_dim(dummy_goal)
  # packed observation, as fed to policy by the environment
  # -- observations from environment like [state; policy goal]
  # -- observations during learning like [state; policy goal; perturbation goal]
  # -- differences in observation shapes are handled by the actor network
  dummy_packed_obs = jnp.concatenate([dummy_state, dummy_goal], axis=-1)
  policy_network = FeedForwardNetwork(
          lambda key: policy.init(key, dummy_packed_obs), policy.apply)
  q_network = FeedForwardNetwork(
          lambda key: critic.init(key, dummy_packed_obs, dummy_action, dummy_goal), critic.apply)

  return ContrastiveNetworks(
      policy_network=policy_network,
      q_network=q_network,
      log_prob=lambda params, actions: params.log_prob(actions),
      sample=lambda params, key: params.sample(seed=key),
      sample_eval=lambda params, key: params.mode())
