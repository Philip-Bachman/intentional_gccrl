"""Haiku modules that output tfd.Distributions."""

from typing import Any, List, Optional, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability
hk_init = hk.initializers
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

_MIN_SCALE = 1e-4
Initializer = hk.initializers.Initializer


class CategoricalHead(hk.Module):
  """Module that produces a categorical distribution with the given number of values."""

  def __init__(
      self,
      num_values: Union[int, List[int]],
      dtype: Optional[Any] = jnp.int32,
      w_init: Optional[Initializer] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._dtype = dtype
    self._logit_shape = num_values
    self._linear = hk.Linear(np.prod(num_values), w_init=w_init)

  def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
    logits = self._linear(inputs)
    if not isinstance(self._logit_shape, int):
      logits = hk.Reshape(self._logit_shape)(logits)
    return tfd.Categorical(logits=logits, dtype=self._dtype)


class TanhTransformedDistribution(tfd.TransformedDistribution):
  """Distribution followed by tanh."""

  def __init__(self, distribution, threshold=.999, validate_args=False):
    """Initialize the distribution.

    Args:
      distribution: The distribution to transform.
      threshold: Clipping value of the action when computing the logprob.
      validate_args: Passed to super class.
    """
    super().__init__(
        distribution=distribution,
        bijector=tfp.bijectors.Tanh(),
        validate_args=validate_args)
    # Computes the log of the average probability distribution outside the
    # clipping range, i.e. on the interval [-inf, -atanh(threshold)] for
    # log_prob_left and [atanh(threshold), inf] for log_prob_right.
    self._threshold = threshold
    inverse_threshold = self.bijector.inverse(threshold)
    # average(pdf) = p/epsilon
    # So log(average(pdf)) = log(p) - log(epsilon)
    log_epsilon = jnp.log(1. - threshold)
    # Those 2 values are differentiable w.r.t. model parameters, such that the
    # gradient is defined everywhere.
    self._log_prob_left = self.distribution.log_cdf(
        -inverse_threshold) - log_epsilon
    self._log_prob_right = self.distribution.log_survival_function(
        inverse_threshold) - log_epsilon

  def log_prob(self, event):
    # Without this clip there would be NaNs in the inner tf.where and that
    # causes issues for some reasons.
    event = jnp.clip(event, -self._threshold, self._threshold)
    # The inverse image of {threshold} is the interval [atanh(threshold), inf]
    # which has a probability of "log_prob_right" under the given distribution.
    return jnp.where(
        event <= -self._threshold, self._log_prob_left,
        jnp.where(event >= self._threshold, self._log_prob_right,
                  super().log_prob(event)))

  def mode(self):
    return self.bijector.forward(self.distribution.mode())

  def entropy(self, seed=None):
    # We return an estimation using a single sample of the log_det_jacobian.
    # We can still do some backpropagation with this estimate.
    return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
        self.distribution.sample(seed=seed), event_ndims=0)

  @classmethod
  def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
    td_properties = super()._parameter_properties(dtype,
                                                  num_classes=num_classes)
    del td_properties['bijector']
    return td_properties


class NormalTanhDistribution(hk.Module):
  """Module that produces a TanhTransformedDistribution distribution."""

  def __init__(self,
               num_dimensions: int,
               min_scale: float = 1e-3,
               w_init: hk_init.Initializer = hk_init.VarianceScaling(
                   0.1, 'fan_in', 'uniform'),
               b_init: hk_init.Initializer = hk_init.Constant(0.),
               rescale: float = 1.0):
    """Initialization.

    Args:
      num_dimensions: Number of dimensions of a distribution.
      min_scale: Minimum standard deviation.
      w_init: Initialization for linear layer weights.
      b_init: Initialization for linear layer biases.
    """
    super().__init__(name='Normal')
    self._min_scale = min_scale
    self._loc_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
    self._scale_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
    self._rescale = tfp.bijectors.Scale(scale=rescale)
    self._pre_tanh_activations = None

  def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
    self._pre_tanh_activations = self._loc_layer(inputs)
    loc = 10. * jax.lax.tanh(self._pre_tanh_activations / 10.)
    scale = jax.nn.softplus(self._scale_layer(inputs)) + self._min_scale
    distribution = tfd.Normal(loc=loc, scale=scale)
    distribution = TanhTransformedDistribution(distribution)
    distribution = tfd.TransformedDistribution(distribution, self._rescale)
    return tfd.Independent(distribution, reinterpreted_batch_ndims=1)


class CategoricalValueHead(hk.Module):
  """Network head that produces a categorical distribution and value."""

  def __init__(
      self,
      num_values: int,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._logit_layer = hk.Linear(num_values)
    self._value_layer = hk.Linear(1)

  def __call__(self, inputs: jnp.ndarray):
    logits = self._logit_layer(inputs)
    value = jnp.squeeze(self._value_layer(inputs), axis=-1)
    return (tfd.Categorical(logits=logits), value)

