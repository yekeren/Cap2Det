
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BIG_NUMBER = 1e10
_SMALL_NUMBER = 1e-10


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


def masked_softmax(data, mask, dim=1):
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

