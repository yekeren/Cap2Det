
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from core import utils

_BIG_NUMBER = 1e8
_SMALL_NUMBER = 1e-8


def _safe_batch_size(tensor):
  """Safely gets the batch size of tensor. 

  Args:
    tensor: a [batch, ...] tensor.

  Returns:
    batch_size: batch size of the tensor.
  """
  batch_size = tensor.get_shape()[0].value
  if batch_size is None:
    batch_size = tf.shape(tensor)[0]
  return batch_size


def sample_all_negative_examples(
    batch_size, name="sample_all_negative_examples"):

  """Samples all negative examples.

  The function returns all True examples in the following matrix:
  / 0, 1, 1, 1 \
  | 1, 0, 1, 1 |
  | 1, 1, 0, 1 |
  \ 1, 1, 1, 0 /
    
  Args:
    batch_size: batch size.

  Returns:
    pos_indices: a [batch] int64 tensor indicateing indices of positive 
      examples.
    neg_indices: a [batch] int64 tensor indicateing indices of negative
      examples.
  """
  with tf.name_scope(name):
    batch_size = tf.cast(batch_size, tf.int64)
    indices = tf.where(
        tf.less(
          tf.diag(tf.fill(tf.expand_dims(batch_size, 0), 1)), 1))
    return indices[:, 0], indices[:, 1]


def sample_random_negative_examples(
    batch_size, 
    negatives_per_sample=1,
    name="sample_random_negative_examples"):

  """Samples random negative examples.

  The function returns random True examples in the following matrix:
  / 0, 1, 1, 1 \
  | 1, 0, 1, 1 |
  | 1, 1, 0, 1 |
  \ 1, 1, 1, 0 /
    
  Args:
    batch_size: batch size.
    negatives_per_sample: number of negatives per each anchor.

  Returns:
    pos_indices: a [batch] int64 tensor indicateing indices of positive
      examples.
    neg_indices: a [batch] int64 tensor indicateing indices of negative 
      examples.
  """
  with tf.name_scope(name):
    batch_size = tf.cast(batch_size, tf.int64)
    pos_indices = tf.tile(tf.range(batch_size), [negatives_per_sample])
    indices = tf.random_uniform(shape=tf.shape(pos_indices), 
        minval=1, maxval=batch_size, dtype=tf.int64)
    neg_indices = tf.mod(pos_indices + indices, batch_size)
    return pos_indices, neg_indices


def triplet_semihard(
    distance_pos, distance_neg, num_captions_pos, num_captions_neg):
  """Processes semihard mining.

  Anchor-negative: the most confusing example from sampled image-caption pair.

  Args:
    distance_pos: a [batch, max_num_captions_pos] float tensor denoting
      distance between anchor image and positive captions.
    distance_neg: a [batch, max_num_captions_neg] float tensor denoting
      distance between anchor image and negative captions.
    num_captions_pos: a [batch] int tensor number of positive captions.
    num_captions_neg: a [batch] int tensor number of negative captions.

  Returns:
    distance_ap: a [batch] float tensor denoting the most confusing (furthest)
      example from the same image-caption pair.
    distance_an: a [batch] float tensor denoting the most confusing (nearest)
      example from the sampled image-caption pair.
  """
  (batch_pos, max_num_captions_pos) = utils.get_tensor_shape(distance_pos)
  (batch_neg, max_num_captions_neg) = utils.get_tensor_shape(distance_neg)

  caption_masks_pos = 1.0 - tf.sequence_mask(
      num_captions_pos, maxlen=max_num_captions_pos, dtype=tf.float32)
  caption_masks_neg = 1.0 - tf.sequence_mask(
      num_captions_neg, maxlen=max_num_captions_neg, dtype=tf.float32)

  distance_ap = tf.reduce_max(
      distance_pos - _BIG_NUMBER * caption_masks_pos, axis=1)
  distance_an = tf.reduce_min(
      distance_neg + _BIG_NUMBER * caption_masks_neg, axis=1)

  return distance_ap, distance_an


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

def triplet_semihard_loss(
    distance_pos, 
    distance_neg, 
    num_captions_pos, 
    num_captions_neg, 
    margin=1.0,
    name="triplet_semihard_loss"):

  """Computes the triplet loss with semi-hard negative mining.

  Args:
    distance_pos: a [batch, max_num_captions] float tensor denoting
      distance between anchor image and positive captions.
    distance_neg: a [batch, max_num_captions] float tensor denoting
      distance between anchor image and negative captions.
    num_captions_pos: a [batch] int tensor number of positive captions.
    num_captions_neg: a [batch] int tensor number of negative captions.
  """
  with tf.name_scope(name):
    distance_ap, distance_an = triplet_semihard(
        distance_pos, distance_neg, num_captions_pos, num_captions_neg)

    losses = tf.maximum(distance_ap - distance_an + margin, 0)
    num_loss_examples = tf.count_nonzero(losses, dtype=tf.float32)

    ratio_optimized_examples = tf.div(
        tf.count_nonzero(tf.less(distance_ap, distance_an), dtype=tf.float32), 
        _SMALL_NUMBER + tf.cast(tf.shape(distance_ap)[0], tf.float32))

    loss = tf.div(
        tf.reduce_sum(losses), _SMALL_NUMBER + num_loss_examples,
        name="triplet_semihard_loss")

  tf.summary.scalar('losses/triplet_semihard_loss', loss)
  tf.summary.scalar('losses/num_loss_examples', num_loss_examples)
  tf.summary.scalar(
      'losses/ratio_optimized_examples', ratio_optimized_examples)

  return loss
