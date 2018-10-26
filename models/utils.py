
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from core import utils
from core import imgproc

_SMALL_NUMBER = 1e-10


def gather_in_batch_captions(
    image_id, num_captions, caption_strings, caption_lengths):

  """Gathers all of the in-batch captions into a caption batch.

  Args:
    image_id: image_id, a [batch] string tensor.
    num_captions: number of captions of each example, a [batch] int tensor.
    caption_strings: caption data, a [batch, max_num_captions, 
      max_caption_length] string tensor.
    caption_lengths: length of each caption, a [batch, max_num_captions] int
      tensor.

  Returns:
    image_ids_gathered: associated image_id of each caption in the new batch, a
      [num_captions_in_batch] string tensor.
    caption_strings_gathered: caption data, a [num_captions_in_batch,
      max_caption_length] string tensor.
    caption_lengths_gathered: length of each caption, a [num_captions_in_batch]
      int tensor.
  """
  (batch, max_num_captions, max_caption_length
   ) = utils.get_tensor_shape(caption_strings)

  # caption_mask denotes the validity of each caption in the flattened batch.
  # caption_mask shape = [batch * max_num_captions], 

  caption_mask = tf.sequence_mask(
      num_captions, maxlen=max_num_captions, dtype=tf.bool)
  caption_mask = tf.reshape(caption_mask, [-1])

  # image_id shape = [batch, max_num_captions].

  image_id = tf.tile(
      tf.expand_dims(image_id, axis=1), [1, max_num_captions])

  # Reshape the tensors to make their first dimensions to be [batch * max_num_captions].

  image_id_reshaped = tf.reshape(image_id, [-1])
  caption_strings_reshaped = tf.reshape(
      caption_strings, [-1, max_caption_length])
  caption_lengths_reshaped= tf.reshape(caption_lengths, [-1])

  # Apply the caption_mask.

  image_ids_gathered = tf.boolean_mask(image_id_reshaped, caption_mask)
  caption_strings_gathered = tf.boolean_mask(
      caption_strings_reshaped, caption_mask)
  caption_lengths_gathered = tf.boolean_mask(
      caption_lengths_reshaped, caption_mask)

  return image_ids_gathered, caption_strings_gathered, caption_lengths_gathered


def _get_expanded_box(box, img_h, img_w, border_ratio):
  """Gets expanded box.

  Args:
    box: a [..., 4] int tensor representing [ymin, xmin, ymax, xmax].
    img_h: image height.
    img_w: image width.
    border_ratio: width of the border in terms of percentage.

  Returns:
    expanded_box: a [..., 4] int tensor with border expanded.
  """
  ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
  (box_h, box_w) = ymax - ymin, xmax - xmin

  border_h = tf.cast(tf.cast(box_h, tf.float32) * border_ratio, tf.int64)
  border_w = tf.cast(tf.cast(box_w, tf.float32) * border_ratio, tf.int64)

  ymin_expanded = tf.maximum(ymin - border_h, 0)
  xmin_expanded = tf.maximum(xmin - border_w, 0)
  ymax_expanded = tf.minimum(ymax + border_h, img_h)
  xmax_expanded = tf.minimum(xmax + border_w, img_w)

  return tf.stack([
      ymin_expanded, xmin_expanded, ymax_expanded, xmax_expanded], axis=-1)


def _get_box_shape(box):
  """Gets the height and width of the box.

  Args:
    box: a [..., 4] int tensor representing [ymin, xmin, ymax, xmax].

  Returns:
    box_h: [...] box height.
    box_w: [...] box width.
  """
  ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
  return ymax - ymin, xmax - xmin


def build_proposal_saliency_fn(func_name, **kwargs):
  """Builds and returns a callable to compute the proposal saliency.

  Args:
    func_name: name of the method.

  Returns:
    a callable that takes `score_map` and `box` as parameters.
  """
  if func_name == 'saliency_sum':
    return imgproc.calc_cumsum_2d

  if func_name == 'saliency_avg':

    def _cumsum_avg(score_map, box):
      ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
      area = tf.expand_dims((ymax - ymin) * (xmax - xmin), axis=-1)
      return tf.div( 
          imgproc.calc_cumsum_2d(score_map, box), 
          _SMALL_NUMBER + tf.cast(area, tf.float32))

    return _cumsum_avg

  if func_name == 'saliency_grad':
    border_ratio = 0.1
    
    def _cumsum_gradient(score_map, box):
      b, n, m, c = utils.get_tensor_shape(score_map)
      _, p, _ = utils.get_tensor_shape(box)

      expanded_box = _get_expanded_box(
          box, img_h=n, img_w=m, border_ratio=0.1)

      (box_h, box_w) = _get_box_shape(box)
      (expanded_box_h, expanded_box_w) = _get_box_shape(expanded_box)

      cumsum = imgproc.calc_cumsum_2d(
          score_map, tf.concat([box, expanded_box], axis=1))

      area = tf.expand_dims(tf.cast(box_h * box_w, tf.float32), axis=-1)
      area_border = tf.expand_dims(
          tf.cast(expanded_box_h * expanded_box_w - box_h * box_w, tf.float32),
          axis=-1)

      avg_val = tf.div(cumsum[:, :p, :], _SMALL_NUMBER + area)
      avg_val_in_border = tf.div(
          cumsum[:, p:, :] - cumsum[:, :p, :],
          _SMALL_NUMBER + area_border)

      return avg_val - avg_val_in_border

    return _cumsum_gradient

  raise ValueError('Invalid func_name {}'.format(func_name))
