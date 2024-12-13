"""Contrastive RL builder."""
import functools
from typing import Callable, Iterator, List, Optional

import jax
import optax
import reverb
from reverb import rate_limiters
import tensorflow as tf
import tree

import acme
from acme import adders
from acme import core
from acme import specs
from acme import types
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers

import contrastive.config as contrastive_config
import contrastive.learning as contrastive_learning
import contrastive.networks as contrastive_networks
import contrastive.utils as contrastive_utils
import contrastive.actors as contrastive_actors



class ContrastiveBuilder(builders.ActorLearnerBuilder):
  """Contrastive RL builder."""

  def __init__(
      self,
      config,
      logger_fn
  ):
    """Creates a contrastive RL learner, a behavior policy and an eval actor.

    Args:
      config: a config with contrastive RL hyperparameters
      logger_fn: a logger factory for the learner
    """
    self._config = config
    self._logger_fn = logger_fn

  def make_learner(
      self,
      random_key,
      networks,
      dataset,
      counter=None
  ):
    print('**************************')
    print('**************************')
    print('***** MAKING LEARNER *****')
    print('**************************')
    print('**************************')
    # Create optimizers
    policy_optimizer = optax.adam(
        learning_rate=self._config.actor_learning_rate, eps=1e-5)
    q_optimizer = optax.adam(learning_rate=self._config.learning_rate, eps=1e-5)
    return contrastive_learning.ContrastiveLearner(
        networks=networks,
        rng=random_key,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        iterator=dataset,
        counter=counter,
        logger=self._logger_fn(),
        config=self._config)

  def make_actor(
      self,
      random_key,
      policy_network,
      variable_source,
      rb_adder = None
  ):
    actor_core = \
      actor_core_lib.batched_feed_forward_to_actor_core(policy_network)
    variable_client = \
      variable_utils.VariableClient(variable_source, 'policy', device='cpu')
    
    _actor = contrastive_actors.ContrastiveGaussianActor(
      actor_core, random_key, variable_client, rb_adder, backend='cpu',
      initially_random=self._config.use_random_actor
    )
    return _actor

  def make_replay_tables(
      self,
      environment_spec,
  ):
    """Create tables to insert data into."""
    samples_per_insert_tolerance = (
        self._config.samples_per_insert_tolerance_rate
        * self._config.samples_per_insert)
    min_replay_traj = self._config.min_replay_size  // self._config.max_episode_steps
    max_replay_traj = self._config.max_replay_size  // self._config.max_episode_steps
    error_buffer = min_replay_traj * samples_per_insert_tolerance
    limiter = rate_limiters.SampleToInsertRatio(
        min_size_to_sample=min_replay_traj,
        samples_per_insert=self._config.samples_per_insert,
        error_buffer=error_buffer)
    rb_signature = adders_reverb.EpisodeAdder.signature(environment_spec, {})
    return [
        reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_traj,
            rate_limiter=limiter,
            signature=rb_signature)
    ]

  def make_dataset_iterator(self,
      replay_client,
      prefetch_size=1,
      device_prefetch=True
    ):
    """Create a dataset iterator to use for learning/updating the agent.
    
    We assume a contrastive environment which provides "packed" observations
    like: [state; goal; latent].
    """
    @tf.function
    def flatten_fn(sample):
      seq_len = tf.shape(sample.data.observation)[0]
      arange = tf.range(seq_len)
      is_future_mask = tf.cast(arange[:, None] < arange[None], tf.float32)
      discount = self._config.discount ** tf.cast(arange[None] - arange[:, None], tf.float32)
      probs = is_future_mask * discount
      # The indexing changes the shape from [seq_len, 1] to [seq_len]
      future_index = tf.random.categorical(logits=tf.math.log(probs),
                                         num_samples=1)[:, 0]
      state = sample.data.observation[:-1, :self._config.obs_dim]
      next_state = sample.data.observation[1:, :self._config.obs_dim]

      # goal that was conditioned on during each episode...
      eps_intent = sample.data.observation[:-1, self._config.obs_dim:]

      # Create the goal observations in three steps.
      # 1. Take all future states (not future goals).
      # 2. Apply obs_to_goal.
      # 3. Sample one of the future states. Note that we don't look for a goal
      # for the final state, because there are no future states.
      future_state = sample.data.observation[:, :self._config.obs_dim]
      future_state = tf.gather(future_state, future_index[:-1])
      new_obs = tf.concat([state, future_state], axis=1)
      new_next_obs = tf.concat([next_state, future_state], axis=1)
      
      transition = types.Transition(
          observation=new_obs,
          action=sample.data.action[:-1],
          reward=sample.data.reward[:-1],
          discount=sample.data.discount[:-1],
          next_observation=new_next_obs,
          extras={
              'state_current': state,
              'state_future': future_state,
              'episode_intent': eps_intent
          })
      # Shift for the transpose_shuffle.
      shift = tf.random.uniform((), 0, seq_len, tf.int32)
      transition = tree.map_structure(lambda t: tf.roll(t, shift, axis=0),
                                      transition)
      return transition

    if self._config.num_parallel_calls:
      num_parallel_calls = self._config.num_parallel_calls
    else:
      num_parallel_calls = tf.data.AUTOTUNE

    def _make_dataset(unused_idx):
      dataset = reverb.TrajectoryDataset.from_table_signature(
          server_address=replay_client.server_address,
          table=self._config.replay_table_name,
          max_in_flight_samples_per_worker=100)
      dataset = dataset.map(flatten_fn)
      # transpose_shuffle
      def _transpose_fn(t):
        dims = tf.range(tf.shape(tf.shape(t))[0])
        perm = tf.concat([[1, 0], dims[2:]], axis=0)
        return tf.transpose(t, perm)
      dataset = dataset.batch(self._config.batch_size, drop_remainder=True)
      dataset = dataset.map(
          lambda transition: tree.map_structure(_transpose_fn, transition))
      dataset = dataset.unbatch()
      # end transpose_shuffle

      dataset = dataset.unbatch()
      return dataset
    dataset = tf.data.Dataset.from_tensors(0).repeat()
    dataset = dataset.interleave(
        map_func=_make_dataset,
        cycle_length=num_parallel_calls,
        num_parallel_calls=num_parallel_calls,
        deterministic=False)

    dataset = dataset.batch(
        self._config.batch_size * self._config.num_sgd_steps_per_step,
        drop_remainder=True)
    @tf.function
    def add_info_fn(data):
      info = reverb.SampleInfo(key=0,
                               probability=0.0,
                               table_size=0,
                               priority=0.0)
      return reverb.ReplaySample(info=info, data=data)
    dataset = dataset.map(add_info_fn, num_parallel_calls=tf.data.AUTOTUNE,
                          deterministic=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    iterator = dataset.as_numpy_iterator()

    if prefetch_size > 1:
      # When working with single GPU we should prefetch to device for
      # efficiency. If running on TPU this isn't necessary as the computation
      # and input placement can be done automatically. For multi-gpu currently
      # the best solution is to pre-fetch to host although this may change in
      # the future.
      device = jax.devices()[0] if device_prefetch else None
      iterator = utils.prefetch(iterator, buffer_size=prefetch_size, device=device)
    return iterator

  def make_adder(self,
      replay_client
  ):
    """Create an adder to record data generated by the actor/environment."""
    return adders_reverb.EpisodeAdder(
        client=replay_client,
        priority_fns={self._config.replay_table_name: None},
        max_sequence_length=self._config.max_episode_steps + 1)
