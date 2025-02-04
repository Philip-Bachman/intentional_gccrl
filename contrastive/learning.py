"""Contrastive RL learner implementation."""
import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Callable

import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import logsumexp
import numpy as np
from jax import random
from jax import lax

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting

import contrastive.config as contrastive_config
import contrastive.networks as contrastive_networks


def global_grad_norm(grads):
    # Extract all leaves (arrays) from the gradient pytree
    leaves, _ = jax.tree_util.tree_flatten(grads)
    # Sum of squares of all elements in all leaves
    sum_of_squares = sum([jnp.sum(jnp.square(g)) for g in leaves])
    return jnp.sqrt(sum_of_squares)


class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  q_optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  target_policy_params: networks_lib.Params
  q_params: networks_lib.Params
  target_q_params: networks_lib.Params
  key: networks_lib.PRNGKey


class ContrastiveLearner(acme.Learner):
  """Contrastive RL learner."""

  _state: TrainingState

  def __init__(
      self,
      networks,
      rng,
      policy_optimizer,
      q_optimizer,
      iterator,
      counter,
      logger,
      config
  ):
    """Initialize the Contrastive RL learner.

    Args:
      networks: Contrastive RL networks.
      rng: a key for random number generation.
      policy_optimizer: the policy optimizer.
      q_optimizer: the Q-function optimizer.
      iterator: iterator over training data. -- replay buffer
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      config: the experiment config file.
    """
    assert logger is not None
    if config.add_mc_to_td:
      assert config.use_td
    self._num_sgd_steps_per_step = config.num_sgd_steps_per_step
    self._obs_dim = config.obs_dim
    self._use_td = config.use_td
    # iterator over training data for actor and critic (ie, replay buffer)
    self._iterator = iterator  
    # stuff for tracking training progress and reporting metrics
    self._counter = counter or counting.Counter()
    self._logger = logger

    def critic_loss(
        q_params,
        policy_params,
        target_q_params,
        transitions,
        key
    ):
      batch_size = transitions.observation.shape[0]
      # Note: We might be able to speed up the computation for some of the
      # baselines to making a single network that returns all the values. This
      # avoids computing some of the underlying representations multiple times.

      I = jnp.eye(batch_size)

      # each (state, policy goal, perturbation goal) tuple corresponds to:
      state = transitions.extras['state_current']      # current state
      pert_goal = transitions.extras['state_future']   # state we perturb towards
      policy_goal = transitions.extras['policy_goal']  # state policy is seeking
      action = transitions.action
      obs_packed = jnp.concatenate([state, policy_goal], axis=1)
      
      # compute logits for use in infonce loss
      logits, _, _ = networks.q_network.apply(q_params, obs_packed, action, pert_goal)

      def loss_fn(_logits):
        loss_nce = optax.softmax_cross_entropy(logits=_logits, labels=I)
        loss_reg = jax.nn.logsumexp(_logits, axis=1)**2
        return loss_nce + 0.01 * loss_reg

      loss = loss_fn(logits)
      loss = jnp.mean(loss)
      correct = (jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1))
      logits_pos = jnp.sum(logits * I) / jnp.sum(I)
      logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
      if len(logits.shape) == 3:
        logsumexp = jax.nn.logsumexp(logits[:, :, 0], axis=1)**2
      else:
        logsumexp = jax.nn.logsumexp(logits, axis=1)**2
      metrics = {
          'categorical_accuracy': jnp.mean(correct),
          'logits_pos': logits_pos,
          'logits_neg': logits_neg,
          'logsumexp': logsumexp.mean(),
      }
      return loss, metrics

    def actor_loss(
        policy_params,
        target_policy_params,
        q_params,
        alpha,
        transitions,
        key,
      ):

      # state       : current state
      # pert_goal   : discounted future state
      # policy_goal : goal of policy that produced this (state, future state) pair
      state = transitions.extras['state_current']
      pert_goal = transitions.extras['state_future']
      policy_goal = transitions.extras['policy_goal']  # use "on-policy" policy goals
      pert_goal_shuffled = jnp.roll(pert_goal, 1, axis=0)

      # train actor 50/50 on intra-episode future states and random states
      train_state = jnp.concatenate([state, state, state], axis=0)
      train_pert_goal = jnp.concatenate([pert_goal, pert_goal_shuffled, policy_goal], axis=0)
      train_policy_goal = jnp.concatenate([policy_goal, policy_goal, policy_goal], axis=0)
      chunk_size = train_state.shape[0] // 3

      obs_packed = jnp.concatenate([train_state, train_policy_goal], axis=-1)
      dist_params = networks.policy_network.apply(
        policy_params, jnp.concatenate([obs_packed, train_pert_goal], axis=-1))
      action = networks.sample(dist_params, key)
      action_log_prob = networks.log_prob(dist_params, action)

      # compute loss for optimizing goal-conditioned actor
      q_action, _, g_repr = networks.q_network.apply(
        q_params, obs_packed, action, train_pert_goal)
      actor_loss = -jnp.diag(q_action) # negative -(Q): maximize Q

      # action entropy loss
      approx_entropy = -action_log_prob

      if config.use_action_entropy:
        actor_loss -= alpha * approx_entropy # negative -(-log prob): maximize entropy

      # split up actor loss into chunks with different meaning
      actor_loss_pert_goal = jnp.mean(actor_loss[:chunk_size])
      actor_loss_pert_goal_shuffled = jnp.mean(actor_loss[chunk_size:(2 * chunk_size)])
      actor_loss_policy_goal = jnp.mean(actor_loss[(2 * chunk_size):])
      loss_sgcrl = 0.5 * (actor_loss_pert_goal + actor_loss_pert_goal_shuffled)
      loss_crl = actor_loss_policy_goal
      blend = 1.0
      actor_loss = (blend * loss_sgcrl) + ((1. - blend) * loss_crl)

      # compute some potentially interesting metrics for goal similarity
      g_repr_pert = g_repr[:chunk_size, :]
      g_repr_policy = g_repr[(2 * chunk_size):, :]
      state_goal_sim = jnp.sum((g_repr_pert * g_repr_policy), axis=1)
      grp_norm_1 = jnp.linalg.norm(g_repr_pert, axis=1)
      grp_norm_2 = jnp.linalg.norm(g_repr_policy, axis=1)
      state_goal_csim = state_goal_sim / (grp_norm_1 * grp_norm_2 + 1e-5)

      metrics = {
          'entropy_mean': jnp.mean(approx_entropy),
          'actor_loss_pert_goal': actor_loss_pert_goal,
          'actor_loss_pert_goal_shuffled': actor_loss_pert_goal_shuffled,
          'actor_loss_policy_goal': actor_loss_policy_goal,
          'actor_state_goal_sim': jnp.mean(state_goal_sim),
          'actor_state_goal_csim': jnp.mean(state_goal_csim),
      }

      return actor_loss, metrics

    # compute gradients for actor and critic
    critic_grad = jax.value_and_grad(critic_loss, has_aux=True)
    actor_grad = jax.value_and_grad(actor_loss, has_aux=True)

    # define the main agent/model update step
    def update_step(
        state, transitions
    ):
      key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)
      alpha = config.entropy_alpha

      ##################################################
      # Compute loss and gradient for critic and actor #
      ##################################################
      (critic_loss, critic_metrics), critic_grads = critic_grad(
          state.q_params, state.policy_params, state.target_q_params,
          transitions, key_critic)
      metrics = critic_metrics
      # ...
      (actor_loss, actor_metrics), actor_grads = actor_grad(
        state.policy_params, state.target_policy_params, state.q_params, 
        alpha, transitions, key_actor)
      metrics.update(actor_metrics)

      #####################################
      # Apply updates to critic and actor #
      #####################################
      clip_critic_norm = optax.clip_by_global_norm(10.0)
      critic_grads_clipped, q_optimizer_state = clip_critic_norm.update(
        critic_grads, state.q_optimizer_state
      )
      critic_update, q_optimizer_state = q_optimizer.update(
        critic_grads_clipped, q_optimizer_state
      )
      q_params = optax.apply_updates(state.q_params, critic_update)
      new_target_q_params = jax.tree_map(lambda x, y: x * (1 - config.tau) + y * config.tau, 
                                         state.target_q_params, q_params)
      # ...
      clip_actor_norm = optax.clip_by_global_norm(1.0)
      actor_grads_clipped, policy_optimizer_state = clip_actor_norm.update(
        actor_grads, state.policy_optimizer_state
      )
      actor_update, policy_optimizer_state = policy_optimizer.update(
          actor_grads_clipped, policy_optimizer_state
      )
      policy_params = optax.apply_updates(state.policy_params, actor_update)
      new_target_policy_params = jax.tree_map(lambda x, y: x * (1 - config.tau) + y * config.tau, 
                                              state.target_policy_params, policy_params)
      
      # get gradient norms for tracking/metrics
      grad_norm_actor = global_grad_norm(actor_grads)
      grad_norm_critic = global_grad_norm(critic_grads)

      # add actor and critic losses to metrics                        
      metrics.update({
          'critic_loss': critic_loss,
          'actor_loss': actor_loss,
          'grad_norm_critic': grad_norm_critic,
          'grad_norm_actor': grad_norm_actor
      })
      
      # update training state
      new_state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          target_policy_params=new_target_policy_params,
          q_params=q_params,
          target_q_params=new_target_q_params,
          key=key
      )    
      return new_state, metrics

    # setup function for performing multiple update steps
    update_step = utils.process_multiple_batches(update_step, config.num_sgd_steps_per_step)
    if config.jit:
      self._update_step = jax.jit(update_step)
    else:
      self._update_step = update_step

    def make_initial_state(
        key
    ):
      """Initialises the training state (parameters and optimiser state)."""
      key_policy, key_q, key = jax.random.split(key, 3)
      # initialize actor and critic params and optimizers
      policy_params = networks.policy_network.init(key_policy)
      policy_optimizer_state = policy_optimizer.init(policy_params)
      q_params = networks.q_network.init(key_q)
      q_optimizer_state = q_optimizer.init(q_params)
      # setup a fresh training state
      state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          target_policy_params=policy_params,
          q_params=q_params,
          target_q_params=q_params,
          key=key
      )
      return state

    # Create initial state.
    self._state = make_initial_state(rng)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online
    # and fill the replay buffer.
    self._timestamp = None

  def step(self):
    with jax.profiler.StepTraceAnnotation('step', step_num=self._counter):
      sample = next(self._iterator)
      transitions = types.Transition(*sample.data)
      self._state, metrics = self._update_step(self._state, transitions) 
    
    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp
    
    # Increment counts and record the current time
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    
    if elapsed_time > 0:
      metrics['steps_per_second'] = (
          self._num_sgd_steps_per_step / elapsed_time)
    else:
      metrics['steps_per_second'] = 0.
    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names):
    variables = {
        'policy': self._state.policy_params,
        'critic': self._state.q_params,
    }
    return [variables[name] for name in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state
