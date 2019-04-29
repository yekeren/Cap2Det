from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import warnings
import cv2
import numpy as np
import tensorflow as tf

_BIG_NUMBER = 1e10
_SMALL_NUMBER = 1e-10


def deprecated(func):
  """A decorator which can be used to mark functions as deprecated.
  
  Args:
    func: the actual function.

  Returns:
    new_func: a wrapping of the actual function `func`.
  """

  @functools.wraps(func)
  def new_func(*args, **kwargs):
    warnings.warn(
        "Function `{}` is deprecated.".format(func.__name__),
        category=DeprecationWarning,
        stacklevel=2)
    tf.logging.warn("Function `{}` is deprecated.".format(func.__name__))
    return func(*args, **kwargs)

  return new_func


def get_tensor_shape(tensor):
  """Gets tensor shape.

  This function uses get_shape() to get static shape, then uses tf.shape() to 
  get unknown dimensions of the tensor.

  Args:
    tensor: a tf.Tensor instance.

  Returns:
    shape: a list of integer of scalar int tensor denoting the tensor shape.
  """
  if not isinstance(tensor, tf.Tensor):
    raise ValueError('The input is not an instance of tf.Tensor.')

  shape_static = tensor.get_shape().as_list()
  shape_dynamic = tf.shape(tensor)

  shape = []
  for i, v in enumerate(shape_static):
    if v is None:
      v = shape_dynamic[i]
    shape.append(v)
  return shape


def masked_maximum(data, mask, dim=1):
  """Computes the axis wise maximum over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the maximum.

  Returns:
    masked_maximums: N-D `Tensor`.
      The maximized dimension is of size 1 after the operation.
  """
  axis_minimums = tf.reduce_min(data, dim, keepdims=True)
  masked_maximums = tf.reduce_max(
      tf.multiply(data - axis_minimums, mask), dim,
      keepdims=True) + axis_minimums
  return masked_maximums


def masked_minimum(data, mask, dim=1):
  """Computes the axis wise minimum over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the minimum.

  Returns:
    masked_minimum: N-D `Tensor`.
      The minimized dimension is of size 1 after the operation.
  """
  axis_maximums = tf.reduce_max(data, dim, keepdims=True)
  masked_minimums = tf.reduce_min(
      tf.multiply(data - axis_maximums, mask), dim,
      keepdims=True) + axis_maximums
  return masked_minimums


def masked_sum(data, mask, dim=1):
  """Computes the axis wise sum over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the sum.

  Returns:
    masked_sum: N-D `Tensor`.
      The summed dimension is of size 1 after the operation.
  """
  return tf.reduce_sum(tf.multiply(data, mask), dim, keepdims=True)


def masked_avg(data, mask, dim=1):
  """Computes the axis wise avg over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the avg.

  Returns:
    masked_avg: N-D `Tensor`.
      The averaged dimension is of size 1 after the operation.
  """
  masked_sums = masked_sum(data, mask, dim)
  masked_avgs = tf.div(
      masked_sums,
      tf.maximum(_SMALL_NUMBER, tf.reduce_sum(mask, dim, keepdims=True)))
  return masked_avgs


def masked_sum_nd(data, mask, dim=1):
  """Computes the axis wise sum over chosen elements.

  Args:
    data: 3-D float `Tensor` of size [n, m, d].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the sum.

  Returns:
    masked_sum: N-D `Tensor`.
      The summed dimension is of size 1 after the operation.
  """
  return tf.reduce_sum(
      tf.multiply(data, tf.expand_dims(mask, axis=-1)), dim, keepdims=True)


def masked_avg_nd(data, mask, dim=1):
  """Computes the axis wise avg over chosen elements.

  Args:
    data: 3-D float `Tensor` of size [n, m, d].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the avg.

  Returns:
    masked_avg: N-D `Tensor`.
      The averaged dimension is of size 1 after the operation.
  """
  masked_sums = masked_sum_nd(data, mask, dim)
  masked_avgs = tf.div(
      masked_sums,
      tf.maximum(
          _SMALL_NUMBER,
          tf.expand_dims(tf.reduce_sum(mask, dim, keepdims=True), axis=-1)))
  return masked_avgs


def masked_softmax(data, mask, dim=-1):
  """Computes the axis wise softmax over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the softmax.

  Returns:
    masked_softmax: 2-D float `Tensor` of size [n, m].
  """
  mask = _BIG_NUMBER * (1.0 - mask)
  return tf.nn.softmax(data - mask, axis=dim)


def masked_argmax(data, mask, dim=1):
  """Computes the axis wise argmax over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the argmax.

  Returns:
    masked_argmax: N-D `Tensor`.
  """
  axis_minimums = tf.reduce_min(data, dim, keepdims=True)
  return tf.argmax(tf.multiply(data - axis_minimums, mask), dim)


def masked_argmin(data, mask, dim=1):
  """Computes the axis wise argmin over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the argmin.

  Returns:
    masked_argmin: N-D `Tensor`.
  """
  axis_maximums = tf.reduce_max(data, dim, keepdims=True)
  return tf.argmin(tf.multiply(data - axis_maximums, mask), dim)


def covariance(x):
  """Computes covariance matrix of data x.

  Args:
    x: 2-D float `Tensor` of size [n, m].

  Returns:
    cov: 2D float `Tensor` of size [n, n].
  """
  x = x - tf.reduce_mean(x, axis=1, keep_dims=True)
  cov = tf.matmul(x, tf.transpose(x)) / tf.to_float(tf.shape(x)[1])
  return cov
