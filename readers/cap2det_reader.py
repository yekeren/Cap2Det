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
  if not isinstance(options, reader_pb2.WSODReader):
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

    feature_dict = {
        InputDataFields.image_id: image_id,
        InputDataFields.num_captions: num_captions,
        InputDataFields.caption_strings: caption_strings,
        InputDataFields.caption_lengths: caption_lengths,
        InputDataFields.concat_caption_string: tokens,
        InputDataFields.concat_caption_length: tf.shape(tokens)[0],
    }

    operations = None
    if options.decode_image:

      with tf.name_scope(OperationNames.decode_image):
        image = tf.image.decode_jpeg(
            parsed[TFExampleDataFields.image_encoded], channels=_IMAGE_CHANNELS)
        if options.HasField("preprocess_options"):
          image, operations = preprocess.preprocess_image_v2(
              image, options.preprocess_options)

        image_height, image_width, _ = utils.get_tensor_shape(image)

        resize_fn = function_builder.build_image_resizer(options.image_resizer)
        image, image_shape = resize_fn(image)
      feature_dict.update({
          InputDataFields.image: image,
          InputDataFields.image_height: image_height,
          InputDataFields.image_width: image_width,
          InputDataFields.image_shape: image_shape,
      })
    # END if options.decode_image:

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

    feature_dict.update({
        InputDataFields.num_proposals: tf.shape(proposals)[0],
        InputDataFields.proposals: proposals,
        InputDataFields.num_objects: tf.shape(object_boxes)[0],
        InputDataFields.object_boxes: object_boxes,
        InputDataFields.object_texts: object_texts,
    })
    return feature_dict

  def _batch_resize_image_fn(examples):

    # Resize image, height and width denote the padding size.

    image = examples[InputDataFields.image]
    _, height, width, channels = utils.get_tensor_shape(image)

    index = tf.random_uniform([],
                              minval=0,
                              maxval=len(options.batch_resize_scale_value),
                              dtype=tf.int32)
    scale_h = scale_w = tf.gather([x for x in options.batch_resize_scale_value],
                                  index)
    new_height = tf.to_int32(tf.round(scale_h * tf.to_float(height)))
    new_width = tf.to_int32(tf.round(scale_w * tf.to_float(width)))

    new_image = tf.image.resize_images(image, tf.stack([new_height, new_width]))
    examples[InputDataFields.image] = new_image

    # Modify the image_shape, height and width denote the image size.

    image_shape = examples[InputDataFields.image_shape]
    height, width, channels = tf.unstack(image_shape, axis=-1)
    new_height = tf.to_int32(tf.round(scale_h * tf.to_float(height)))
    new_width = tf.to_int32(tf.round(scale_w * tf.to_float(width)))

    new_image_shape = tf.stack([new_height, new_width, channels], axis=-1)
    examples[InputDataFields.image_shape] = new_image_shape

    return examples

  def _batch_scale_box_fn(examples):
    (image, image_shape, object_boxes,
     proposal_boxes) = (examples[InputDataFields.image],
                        examples[InputDataFields.image_shape],
                        examples[InputDataFields.object_boxes],
                        examples[InputDataFields.proposals])

    _, pad_h, pad_w, _ = utils.get_tensor_shape(image)
    img_h, img_w, _ = tf.unstack(image_shape, axis=-1)

    def _scale_box(box):
      ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
      ymin = ymin * tf.to_float(tf.expand_dims(img_h,
                                               axis=-1)) / tf.to_float(pad_h)
      xmin = xmin * tf.to_float(tf.expand_dims(img_w,
                                               axis=-1)) / tf.to_float(pad_w)
      ymax = ymax * tf.to_float(tf.expand_dims(img_h,
                                               axis=-1)) / tf.to_float(pad_h)
      xmax = xmax * tf.to_float(tf.expand_dims(img_w,
                                               axis=-1)) / tf.to_float(pad_w)
      return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    examples[InputDataFields.object_boxes] = _scale_box(object_boxes)
    examples[InputDataFields.proposals] = _scale_box(proposal_boxes)

    return examples

  def _filter_fn(examples):
    #image_id = examples[InputDataFields.image_id]
    #image_id = tf.string_to_number(image_id, out_type=tf.int64)

    #numer, denom = options.shard_indicator.split('/')
    #assert numer.isdigit() and denom.isdigit()

    #numer, denom = int(numer), int(denom)
    #assert 0 <= numer < denom

    #return tf.equal(tf.mod(image_id, denom), numer)
    image_id = examples[InputDataFields.image_id]

    numer, denom = options.shard_indicator.split('/')
    assert numer.isdigit() and denom.isdigit()

    numer, denom = int(numer), int(denom)
    assert 0 <= numer < denom

    hash_bucket = tf.strings.to_hash_bucket(image_id, num_buckets=denom)
    return tf.equal(hash_bucket, numer)

  def _input_fn():
    """Returns a python dictionary.

    Returns:
      a dataset that can be fed to estimator.
    """
    input_pattern = [elem for elem in options.input_pattern]
    files = tf.data.Dataset.list_files(
        input_pattern, shuffle=options.is_training)
    dataset = files.interleave(
        tf.data.TFRecordDataset, cycle_length=options.interleave_cycle_length)
    if options.is_training:
      dataset = dataset.repeat().shuffle(options.shuffle_buffer_size)
    dataset = dataset.map(
        map_func=_parse_fn, num_parallel_calls=options.map_num_parallel_calls)
    if options.shard_indicator:
      dataset = dataset.filter(predicate=_filter_fn)

    padded_shapes = {
        InputDataFields.image_id: [],
        InputDataFields.num_captions: [],
        InputDataFields.caption_strings: [None, None],
        InputDataFields.caption_lengths: [None],
        InputDataFields.num_proposals: [],
        InputDataFields.proposals: [options.max_num_proposals, 4],
        InputDataFields.num_objects: [],
        InputDataFields.object_boxes: [None, 4],
        InputDataFields.object_texts: [None],
        InputDataFields.concat_caption_string: [None],
        InputDataFields.concat_caption_length: [],
    }
    if options.decode_image:
      padded_shapes.update({
          InputDataFields.image: [None, None, _IMAGE_CHANNELS],
          InputDataFields.image_height: [],
          InputDataFields.image_width: [],
          InputDataFields.image_shape: [3],
      })

    dataset = dataset.padded_batch(
        options.batch_size, padded_shapes=padded_shapes, drop_remainder=True)

    # Randomly resize image according to the batch scale values.

    if options.decode_image and len(options.batch_resize_scale_value) > 0:
      dataset = dataset.map(
          map_func=_batch_resize_image_fn, num_parallel_calls=1)

    # Scale the proposal and object boxes according to the padding facts.

    if options.decode_image:
      dataset = dataset.map(map_func=_batch_scale_box_fn, num_parallel_calls=1)

    dataset = dataset.prefetch(options.prefetch_buffer_size)
    return dataset

  return _input_fn
