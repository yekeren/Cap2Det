from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from core.standard_fields import InputDataFields
from core.standard_fields import TFExampleDataFields
from core.standard_fields import OperationNames
from core import preprocess
from protos import reader_pb2


def _parse_captions(tokens, offsets, lengths, max_caption_length=20):
  """Parses and pads captions.

  Args:
    tokens: a [num_tokens] tf.string tensor denoting token buffer.
    offsets: a [num_captions] tf.int64 tensor, denoting the offset of each
      caption in the token buffer.
    lengths: a [num_captions] tf.int64 tensor, denoting the length of each
      caption.
    max_caption_length: a scalar tensor denoting the maximum caption length.

  Returns:
    num_captions: number of captions after padding.
    caption_strings: [num_captions, max_caption_length] tf.string tensor.
    caption_lengths: [num_captions] tf.int64 tensor.
  """
  num_offsets = tf.shape(offsets)[0]
  num_lengths = tf.shape(lengths)[0]

  assert_op = tf.Assert(
      tf.equal(num_offsets, num_lengths),
      ["Not equal: num_offsets and num_lengths", num_offsets, num_lengths])

  with tf.control_dependencies([assert_op]):
    num_captions = num_offsets

    i = tf.constant(0)
    caption_strings = tf.constant(
        "", dtype=tf.string, shape=[0, max_caption_length])
    caption_lengths = tf.constant(0, dtype=tf.int64, shape=[0])

    def _body(i, caption_strings, caption_lengths):
      """Executes the while loop body.

      Note, this function trims or pads captions to make them the same lengths.

      Args:
        i: index of both offsets/lengths tensors.
        caption_strings: aggregated caption strings tensor.
        caption_lengths: aggregated caption lengths tensor.
      """
      offset = offsets[i]
      length = tf.minimum(lengths[i], max_caption_length)

      pad = tf.fill(tf.expand_dims(max_caption_length - length, axis=0), "")
      caption = tokens[offset:offset + length]
      caption = tf.concat([caption, pad], axis=0)
      caption_strings = tf.concat(
          [caption_strings, tf.expand_dims(caption, 0)], axis=0)
      caption_lengths = tf.concat(
          [caption_lengths, tf.expand_dims(length, 0)], axis=0)
      return i + 1, caption_strings, caption_lengths

    cond = lambda i, unused_strs, unused_lens: tf.less(i, num_captions)
    (_, caption_strings, caption_lengths) = tf.while_loop(
        cond,
        _body,
        loop_vars=[i, caption_strings, caption_lengths],
        shape_invariants=[
            i.get_shape(),
            tf.TensorShape([None, max_caption_length]),
            tf.TensorShape([None])
        ])

  return num_captions, caption_strings, caption_lengths


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
    }
    parsed = tf.parse_single_example(
        example, example_fmt, name=OperationNames.parse_single_example)

    image_id = parsed[TFExampleDataFields.image_id]

    with tf.name_scope(OperationNames.decode_image):
      image = tf.image.decode_jpeg(
          parsed[TFExampleDataFields.image_encoded],
          channels=options.image_channels)
      if options.HasField("preprocess_options"):
        image = preprocess.preprocess(image, options.preprocess_options)
      image = tf.image.resize_images(
          image, [options.image_height, options.image_width])

    with tf.name_scope(OperationNames.decode_caption):
      tokens = tf.sparse_tensor_to_dense(
          parsed[TFExampleDataFields.caption_string], default_value="")
      offsets = tf.sparse_tensor_to_dense(
          parsed[TFExampleDataFields.caption_offset], default_value=0)
      lengths = tf.sparse_tensor_to_dense(
          parsed[TFExampleDataFields.caption_length], default_value=0)

      (num_captions, caption_strings, caption_lengths) = _parse_captions(
          tokens,
          offsets,
          lengths,
          max_caption_length=options.max_caption_length)

    feature_dict = {
        InputDataFields.image: image,
        InputDataFields.image_id: image_id,
        InputDataFields.num_captions: num_captions,
        InputDataFields.caption_strings: caption_strings,
        InputDataFields.caption_lengths: caption_lengths,
    }
    return feature_dict

  def _input_fn():
    """Returns a python dictionary.

    Returns:
      a dataset that can be fed to estimator.
    """
    files = tf.data.Dataset.list_files(options.input_pattern)
    dataset = files.interleave(
        tf.data.TFRecordDataset, cycle_length=options.interleave_cycle_length)
    if options.is_training:
      dataset = dataset.repeat().shuffle(options.shuffle_buffer_size)
    dataset = dataset.map(
        map_func=_parse_fn, num_parallel_calls=options.map_num_parallel_calls)
    dataset = dataset.padded_batch(
        options.batch_size,
        padded_shapes={
            InputDataFields.image:
            [options.image_height, options.image_width, options.image_channels],
            InputDataFields.image_id: [],
            InputDataFields.num_captions: [],
            InputDataFields.caption_strings: [None, options.max_caption_length],
            InputDataFields.caption_lengths: [None],
        },
        drop_remainder=True)
    dataset = dataset.prefetch(options.prefetch_buffer_size)
    return dataset

  return _input_fn
