from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from protos import preprocess_pb2


def random_crop(image, random_crop_min_scale):
  """Random crops image according to the minimum scale requirement.

  Args:
    image: a [height, width, 3] uint8 image tensor.
    random_crop_min_scale: minimum scale.

  Returns:
    image: a [crop_height, crop_width, 3] float tensor.
  """
  height, width = tf.shape(image)[0], tf.shape(image)[1]

  min_height = tf.cast(
      tf.round(tf.cast(height, tf.float32) * random_crop_min_scale), tf.int32)
  min_width = tf.cast(
      tf.round(tf.cast(width, tf.float32) * random_crop_min_scale), tf.int32)

  target_height = tf.random_uniform(
      shape=[], dtype=tf.int32, minval=min_height, maxval=height + 1)
  target_width = tf.random_uniform(
      shape=[], dtype=tf.int32, minval=min_width, maxval=width + 1)

  offset_height = tf.random_uniform(
      shape=[], dtype=tf.int32, minval=0, maxval=height + 1 - target_height)
  offset_width = tf.random_uniform(
      shape=[], dtype=tf.int32, minval=0, maxval=width + 1 - target_width)

  image = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                        target_height, target_width)
  return image


def preprocess_func(image, flip_left_right):
  """Preprocesses an image.

  Args:
    image: A [height, width, 3] uint8 tensor.
    flip_left_right: A boolean scalar tensor.
  """
  image = tf.cond(
      flip_left_right,
      true_fn=lambda: tf.image.flip_left_right(image),
      false_fn=lambda: image)
  return image


def preprocess_image_v2(image, options):
  """Get preprocess function based on options.

  Args:
    image: A [height, width, 3] uint8 tensor.
    options: An instance of PreprocessOptions.

  Returns:
    preprocessed_image: A [height, width, 3] uint8 tensor.

  Raises:
    ValueError: If the options is invalid.
  """
  if not isinstance(options, preprocess_pb2.Preprocess):
    raise ValueError('Options has to be an instance of Preprocess.')

  flip_left_right = tf.less(
      tf.random_uniform(shape=[]), options.random_flip_left_right_prob)
  operations = {'flip_left_right': flip_left_right}

  preprocessed_image = preprocess_func(image, flip_left_right)

  return preprocessed_image, operations


def preprocess_image(image, options):
  """Preprocesses an image.

  Args:
    image: a [height, width, 3] uint8 tensor.
    options: an instance of PreprocessOptions.

  Returns:
    preprocessed_image: a [height, width, 3] uint8 tensor.

  Raises:
    ValueError: if options is invalid
  """
  if not isinstance(options, preprocess_pb2.Preprocess):
    raise ValueError('Options has to be an instance of Preprocess.')

  # Change brightness.
  # Note: it has to be applied on uint8 image.

  image = tf.cond(
      tf.less(tf.random_uniform(shape=[]), options.random_brightness_prob),
      true_fn=lambda: tf.image.random_brightness(
        image, max_delta=options.random_brightness_max_delta),
      false_fn=lambda: image)

  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  # Change contrast.

  image = tf.cond(
      tf.less(tf.random_uniform(shape=[]), options.random_contrast_prob),
      true_fn=lambda: tf.image.random_contrast(image,
        lower=options.random_contrast_lower,
        upper=options.random_contrast_upper),
      false_fn=lambda: image)

  # Change hue.

  image = tf.cond(
      tf.less(tf.random_uniform(shape=[]), options.random_hue_prob),
      true_fn=lambda: tf.image.random_hue(
        image, max_delta=options.random_hue_max_delta),
      false_fn=lambda: image)

  # Change saturation.

  image = tf.cond(
      tf.less(tf.random_uniform(shape=[]), options.random_saturation_prob),
      true_fn=lambda: tf.image.random_saturation(image,
        lower=options.random_saturation_lower,
        upper=options.random_saturation_upper),
      false_fn=lambda: image)

  # Flip left-right.

  image = tf.cond(
      tf.less(tf.random_uniform(shape=[]), options.random_flip_left_right_prob),
      true_fn=lambda: tf.image.flip_left_right(image),
      false_fn=lambda: image)

  # Random crop.

  image = tf.cond(
      tf.less(tf.random_uniform(shape=[]), options.random_crop_prob),
      true_fn=lambda: random_crop(image, options.random_crop_min_scale),
      false_fn=lambda: image)

  return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)


def parse_texts(tokens, offsets, lengths):
  """Parses and pads texts.

  Args:
    tokens: a [num_tokens] tf.string tensor denoting token buffer.
    offsets: a [num_texts] tf.int64 tensor, denoting the offset of each
      text in the token buffer.
    lengths: a [num_texts] tf.int64 tensor, denoting the length of each
      text.

  Returns:
    num_texts: number of texts after padding.
    text_strings: [num_texts, max_text_length] tf.string tensor.
    text_lengths: [num_texts] tf.int64 tensor.
  """
  max_text_length = tf.maximum(tf.reduce_max(lengths), 0)

  num_offsets = tf.shape(offsets)[0]
  num_lengths = tf.shape(lengths)[0]

  assert_op = tf.Assert(
      tf.equal(num_offsets, num_lengths),
      ["Not equal: num_offsets and num_lengths", num_offsets, num_lengths])

  with tf.control_dependencies([assert_op]):
    num_texts = num_offsets

    i = tf.constant(0)
    text_strings = tf.fill(tf.stack([0, max_text_length], axis=0), "")
    text_lengths = tf.constant(0, dtype=tf.int64, shape=[0])

    def _body(i, text_strings, text_lengths):
      """Executes the while loop body.

      Note, this function trims or pads texts to make them the same lengths.

      Args:
        i: index of both offsets/lengths tensors.
        text_strings: aggregated text strings tensor.
        text_lengths: aggregated text lengths tensor.
      """
      offset = offsets[i]
      length = lengths[i]

      pad = tf.fill(tf.expand_dims(max_text_length - length, axis=0), "")
      text = tokens[offset:offset + length]
      text = tf.concat([text, pad], axis=0)
      text_strings = tf.concat([text_strings, tf.expand_dims(text, 0)], axis=0)
      text_lengths = tf.concat(
          [text_lengths, tf.expand_dims(length, 0)], axis=0)
      return i + 1, text_strings, text_lengths

    cond = lambda i, unused_strs, unused_lens: tf.less(i, num_texts)
    (_, text_strings, text_lengths) = tf.while_loop(
        cond,
        _body,
        loop_vars=[i, text_strings, text_lengths],
        shape_invariants=[
            i.get_shape(),
            tf.TensorShape([None, None]),
            tf.TensorShape([None])
        ])

  return num_texts, text_strings, text_lengths
