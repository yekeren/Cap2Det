
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from core import utils


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
