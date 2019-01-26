from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from core.standard_fields import InputDataFields
from core.standard_fields import TFExampleDataFields
from core.standard_fields import OperationNames
from core import preprocess
from core import utils
from core import box_utils
from core import builder as function_builder
from protos import reader_pb2

_IMAGE_CHANNELS = 3


def get_input_fn(options):
  """Returns a function that generate input examples.

  Args:
    options: an instance of reader_pb2.Reader.

  Returns:
    input_fn: a callable that returns a dataset.
  """
  if not isinstance(options, reader_pb2.Reader):
    raise ValueError('options has to be an instance of Reader.')

  def _parse_fn(example):
    """Parses tf::Example proto.

    Args:
      example: a tf::Example proto.

    Returns:
      feature_dict: a dict mapping from names to tensors.
    """
    example_fmt = {
        TFExampleDataFields.image_id: tf.FixedLenFeature((), tf.string),
        TFExampleDataFields.image_encoded: tf.FixedLenFeature((), tf.string),
        TFExampleDataFields.caption_string: tf.VarLenFeature(tf.string),
        TFExampleDataFields.caption_offset: tf.VarLenFeature(tf.int64),
        TFExampleDataFields.caption_length: tf.VarLenFeature(tf.int64),
        TFExampleDataFields.object_box_ymin: tf.VarLenFeature(tf.float32),
        TFExampleDataFields.object_box_xmin: tf.VarLenFeature(tf.float32),
        TFExampleDataFields.object_box_ymax: tf.VarLenFeature(tf.float32),
        TFExampleDataFields.object_box_xmax: tf.VarLenFeature(tf.float32),
        TFExampleDataFields.object_label: tf.VarLenFeature(tf.int64),
        TFExampleDataFields.object_text: tf.VarLenFeature(tf.string),
        TFExampleDataFields.proposal_box_ymin: tf.VarLenFeature(tf.float32),
        TFExampleDataFields.proposal_box_xmin: tf.VarLenFeature(tf.float32),
        TFExampleDataFields.proposal_box_ymax: tf.VarLenFeature(tf.float32),
        TFExampleDataFields.proposal_box_xmax: tf.VarLenFeature(tf.float32),
    }
    parsed = tf.parse_single_example(
        example, example_fmt, name=OperationNames.parse_single_example)

    # Basic image information.

    image_id = parsed[TFExampleDataFields.image_id]

    operations = None
    with tf.name_scope(OperationNames.decode_image):
      image = tf.image.decode_jpeg(
          parsed[TFExampleDataFields.image_encoded], channels=_IMAGE_CHANNELS)
      if options.HasField("preprocess_options"):
        image, operations = preprocess.preprocess_image_v2(
            image, options.preprocess_options)

      resize_fn = function_builder.build_image_resizer(options.image_resizer)
      image, image_shape = resize_fn(image)

    # Caption annotations.

    with tf.name_scope(OperationNames.decode_caption):
      tokens = tf.sparse_tensor_to_dense(
          parsed[TFExampleDataFields.caption_string], default_value="")
      offsets = tf.sparse_tensor_to_dense(
          parsed[TFExampleDataFields.caption_offset], default_value=0)
      lengths = tf.sparse_tensor_to_dense(
          parsed[TFExampleDataFields.caption_length], default_value=0)

      (num_captions, caption_strings, caption_lengths) = preprocess.parse_texts(
          tokens, offsets, lengths)

    # Region proposal annotations.

    with tf.name_scope(OperationNames.decode_proposal):
      bbox_decoder = tf.contrib.slim.tfexample_decoder.BoundingBox(
          prefix=TFExampleDataFields.proposal_box + '/')
      proposals = bbox_decoder.tensors_to_item(parsed)
      if options.is_training and options.shuffle_proposals:
        proposals = tf.random_shuffle(proposals)
      proposals = proposals[:options.max_num_proposals]

      if operations is not None:
        proposals = tf.cond(
            operations['flip_left_right'],
            true_fn=lambda: box_utils.flip_left_right(proposals),
            false_fn=lambda: proposals)

    # Bounding box annotations.

    with tf.name_scope(OperationNames.decode_bbox):
      bbox_decoder = tf.contrib.slim.tfexample_decoder.BoundingBox(
          prefix=TFExampleDataFields.object_box + '/')
      object_boxes = bbox_decoder.tensors_to_item(parsed)
      text_decoder = tf.contrib.slim.tfexample_decoder.Tensor(
          TFExampleDataFields.object_text, default_value='')
      object_texts = text_decoder.tensors_to_item(parsed)

      if operations is not None:
        object_boxes = tf.cond(
            operations['flip_left_right'],
            true_fn=lambda: box_utils.flip_left_right(object_boxes),
            false_fn=lambda: object_boxes)

    feature_dict = {
        InputDataFields.image: image,
        InputDataFields.image_shape: image_shape,
        InputDataFields.image_id: image_id,
        InputDataFields.num_captions: num_captions,
        InputDataFields.caption_strings: caption_strings,
        InputDataFields.caption_lengths: caption_lengths,
        InputDataFields.num_proposals: tf.shape(proposals)[0],
        InputDataFields.proposals: proposals,
        InputDataFields.num_objects: tf.shape(object_boxes)[0],
        InputDataFields.object_boxes: object_boxes,
        InputDataFields.object_texts: object_texts,
    }
    return feature_dict

  def _batch_resize_fn(examples):
    image = examples[InputDataFields.image]
    batch, height, width, channels = utils.get_tensor_shape(image)

    scale_h = scale_w = tf.random_uniform(
        [],
        minval=options.batch_resize_scale_lower,
        maxval=options.batch_resize_scale_upper,
        dtype=tf.float32)
    #scale_w = tf.random_uniform([],
    #                            minval=options.batch_resize_scale_lower,
    #                            maxval=options.batch_resize_scale_upper,
    #                            dtype=tf.float32)
    new_height = tf.to_int32(tf.round(scale_h * tf.to_float(height)))
    new_width = tf.to_int32(tf.round(scale_w * tf.to_float(width)))

    new_image = tf.image.resize_images(image, tf.stack([new_height, new_width]))
    examples[InputDataFields.image] = new_image
    return examples

  def _input_fn():
    """Returns a python dictionary.

    Returns:
      a dataset that can be fed to estimator.
    """
    files = tf.data.Dataset.list_files(
        options.input_pattern, shuffle=options.is_training)
    dataset = files.interleave(
        tf.data.TFRecordDataset, cycle_length=options.interleave_cycle_length)
    if options.is_training:
      dataset = dataset.repeat().shuffle(options.shuffle_buffer_size)
    dataset = dataset.map(
        map_func=_parse_fn, num_parallel_calls=options.map_num_parallel_calls)

    height = width = None

    dataset = dataset.padded_batch(
        options.batch_size,
        padded_shapes={
            InputDataFields.image: [height, width, _IMAGE_CHANNELS],
            InputDataFields.image_id: [],
            InputDataFields.image_shape: [3],
            InputDataFields.num_captions: [],
            InputDataFields.caption_strings: [None, None],
            InputDataFields.caption_lengths: [None],
            InputDataFields.num_proposals: [],
            InputDataFields.proposals: [options.max_num_proposals, 4],
            InputDataFields.num_objects: [],
            InputDataFields.object_boxes: [None, 4],
            InputDataFields.object_texts: [None],
        },
        drop_remainder=True)
    if options.batch_resize_scale:
      dataset = dataset.map(map_func=_batch_resize_fn, num_parallel_calls=1)

    dataset = dataset.prefetch(options.prefetch_buffer_size)
    return dataset

  return _input_fn
