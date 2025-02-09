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
    key, key_a, key_s = jax.random.split(key, 3)
    return sample_fn(networks.policy_network.apply(params, key_a, obs), key_s)
  return apply_and_sample


def hk_linear_layer(n_hidden, init_scale=1.0):
  w_init = hk.initializers.VarianceScaling(init_scale, 'fan_avg', 'uniform')
  return hk.Linear(n_hidden, w_init=w_init)


class MlpLayer(hk.Module):
  """Simple haiku compatible mlp layer -- basic dense or residualish."""
  def __init__(self,
               n_hidden: int,
               layer_type: str
               ):
    assert (layer_type in ['dense', 'residual'])
    super().__init__()
    self.layer_type = layer_type
    block_layers = []

    if layer_type == 'dense':
      # dense block like: x = relu(ln(linear(x)))
      block_layers.append(hk_linear_layer(n_hidden, 1.0))
      block_layers.append(hk.LayerNorm(-1, True, True))
      block_layers.append(jax.nn.relu)
    else:
      # residual block like: x = x + linear(relu(linear(ln(x))))
      block_layers.append(hk.LayerNorm(-1, True, True))
      block_layers.append(hk_linear_layer(n_hidden, 1.0))
      block_layers.append(jax.nn.relu)
      block_layers.append(hk_linear_layer(n_hidden, 1.0))
    self.block = hk.Sequential(block_layers)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    if self.layer_type == 'dense':
      x = self.block(x)
    else:
      x = x + self.block(x)
    return x


def make_mlp(
    hidden_layer_sizes,
    out_size=None,
    out_layer=None,
    simba=True
):
  # user should provide either out_size or out_layer
  assert (out_size is None) or (out_layer is None)
  assert not ((out_size is None) and (out_layer is None))
  # construct the mlp
  layer_list = []
  if simba:
    # start with linear transform and layernorm
    layer_list.append(hk_linear_layer(hidden_layer_sizes[0], 1.0))
  # add hidden layers (same for actor and critic)
  for l_sz in hidden_layer_sizes:
    if simba:
      layer_list.append(MlpLayer(l_sz, layer_type='residual'))
    else:
      layer_list.append(MlpLayer(l_sz, layer_type='dense'))
  if simba:
    layer_list.append(hk.LayerNorm(-1, True, True))
  # add final layers (different for actor and critic)
  if (out_layer is None):
    layer_list.append(hk_linear_layer(out_size, 0.1))
  else:
    layer_list.append(out_layer)
  # mash layers together into a haiku network
  mlp = hk.Sequential(layer_list)
  return mlp


def make_networks(
    spec,
    obs_dim,
    repr_dim = 64,
    hidden_layer_sizes = (256, 256),
    actor_min_std = 1e-2,
    use_image_obs = False):
  """Creates networks used by the agent."""
  assert (not use_image_obs)  # TODO: patch things up to handle image observations
  action_dim = np.prod(spec.actions.shape, dtype=int)
  TORSO = networks_lib.AtariTorso

  def _info_fuzz(x, blend):
    rng_key = hk.next_rng_key()
    fuzz = 0.1 * jax.random.normal(rng_key, shape=x.shape, dtype=x.dtype)
    x_fuzz = (blend * x) + ((1 - blend) * fuzz)
    return x_fuzz

  def _repr_fn(obs_packed, action, pert_goal):
    # obs_packed : should contain current state, goal, and "task mask"
    # action     : should contain action
    # pert_goal  : state we want to predict in the future
    state = obs_packed[:, :obs_dim]
    goal = obs_packed[:, obs_dim:(2 * obs_dim)]
    mask = obs_packed[:, (2 * obs_dim):]

    # encoder for (state, action, policy goal)
    sag_encoder = make_mlp(hidden_layer_sizes, out_size=repr_dim,
                           out_layer=None)
    sag_repr = sag_encoder(jnp.concatenate([state, 0. * _info_fuzz(goal, 0.),
                                            mask, action], axis=-1))

    # encoder for perturbation goals
    g_encoder = make_mlp(hidden_layer_sizes, out_size=repr_dim,
                         out_layer=None)
    genc_input = jnp.concatenate([mask, mask * pert_goal], axis=1)
    g_repr = g_encoder(genc_input)
    return sag_repr, g_repr

  def _combine_repr(sag_repr, g_repr):
    return jax.numpy.einsum('ik,jk->ij', sag_repr, g_repr)

  def _critic_fn(obs_packed, action, pert_goal):
    sag_repr, g_repr = \
      _repr_fn(obs_packed, action, pert_goal)
    critic_val = _combine_repr(sag_repr, g_repr)
    return critic_val, sag_repr, g_repr

  def _actor_fn(obs_packed):
    # input like [state; goal; mask]
    state = obs_packed[:, :obs_dim]
    goal = obs_packed[:, obs_dim:(2 * obs_dim)]
    mask = obs_packed[:, (2 * obs_dim):(3 * obs_dim)]    

    in_dim = obs_packed.shape[1]
    assert (in_dim == (3 * obs_dim)) or (in_dim == (4 * obs_dim))
    if in_dim == (3 * obs_dim):
      pert_goal = goal
    else:
      pert_goal = obs_packed[:, (3 * obs_dim):]
    obs_packed = jnp.concatenate([state, 0. * _info_fuzz(goal, 0.),
                                  mask, pert_goal], axis=1)

    # full packed input to actor like:
    # -- [state; goal; mask; pert_goal]
    dist_layer = NormalTanhDistribution(
      action_dim, min_scale=actor_min_std, rescale=0.99)
    network = make_mlp(hidden_layer_sizes, out_size=None,
                       out_layer=dist_layer)
    return network(obs_packed)

  policy = hk.transform(_actor_fn)
  critic = hk.transform(_critic_fn)

  # create dummy observations and actions to create network parameters.
  # -- it's important to note that the "observation" expected here is a
  #    "packed" observation that includes both a current environment state
  #    and a future goal state of the same form as the current state.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)       # obs is like [state; goal]
  dummy_state = utils.zeros_like(dummy_obs[:obs_dim])
  dummy_goal = utils.zeros_like(dummy_obs[obs_dim:(2 * obs_dim)])
  dummy_mask = utils.zeros_like(dummy_obs[(2 * obs_dim):])
  # ...  
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_state = utils.add_batch_dim(dummy_state)
  dummy_goal = utils.add_batch_dim(dummy_goal)
  dummy_mask = utils.add_batch_dim(dummy_mask)

  # packed observation, as fed to policy by the environment
  # -- observations from environment like [state; policy goal]
  # -- observations during learning like [state; policy goal; perturbation goal]
  # -- differences in observation shapes are handled by the actor network
  dummy_packed_obs = jnp.concatenate([dummy_state, dummy_goal, dummy_mask], axis=-1)
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
