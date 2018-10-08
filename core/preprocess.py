
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

  target_height = tf.random_uniform(shape=[], 
      dtype=tf.int32, minval=min_height, maxval=height + 1)
  target_width = tf.random_uniform(shape=[], 
      dtype=tf.int32, minval=min_width, maxval=width + 1)

  offset_height = tf.random_uniform(shape=[], 
      dtype=tf.int32, minval=0, maxval=height + 1 - target_height)
  offset_width = tf.random_uniform(shape=[], 
      dtype=tf.int32, minval=0, maxval=width + 1 - target_width)

  image = tf.image.crop_to_bounding_box(image, 
      offset_height, offset_width, target_height, target_width)
  return image


def preprocess(image, options):
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
      true_fn=lambda: tf.image.flip_left_right(image), false_fn=lambda: image)

  # Random crop.

  image = tf.cond(
      tf.less(tf.random_uniform(shape=[]), options.random_crop_prob),
      true_fn=lambda: random_crop(image, options.random_crop_min_scale), 
      false_fn=lambda: image)

  return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)
