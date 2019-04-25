from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from core.standard_fields import InputDataFields
from core.standard_fields import TFExampleDataFields
from protos import reader_pb2


class TFExampleDataFields(object):
  """Names of the fields of the tf.train.Example."""
  image_id = "image_id"
  image_feature = 'feature/img/value'
  roi_feature = 'feature/roi/value'
  roi_feature_size = 'feature/roi/length'
  slogan_text_string = 'ocr/text/string'
  slogan_text_offset = 'ocr/text/offset'
  slogan_text_length = 'ocr/text/length'
  groundtruth_text_string = 'label/text/string'
  groundtruth_text_offset = 'label/text/offset'
  groundtruth_text_length = 'label/text/length'
  question_text_string = 'question/text/string'
  question_text_offset = 'question/text/offset'
  question_text_length = 'question/text/length'


class InputDataFields(object):
  """Names of the input tensors."""
  image_id = 'image_id'
  image_feature = 'image_feature'
  roi_feature = 'feature/roi/value'
  roi_feature_size = 'feature/roi/length'
  slogan_text_size = 'slogan_text_size'
  slogan_text_string = 'slogan_text_string'
  slogan_text_length = 'slogan_text_length'
  groundtruth_text_size = 'groundtruth_text_size'
  groundtruth_text_string = 'groundtruth_text_string'
  groundtruth_text_length = 'groundtruth_text_length'
  question_text_size = 'question_text_size'
  question_text_string = 'question_text_string'
  question_text_length = 'question_text_length'


_OP_PARSE_SINGLE_EXAMPLE = 'reader/op_parse_single_example'
_OP_DECODE_IMAGE = 'reader/op_decode_image'
_OP_DECODE_SLOGAN = 'reader/op_decode_slogan'
_OP_DECODE_GROUNDTRUTH = 'reader/op_decode_groundtruth'
_OP_DECODE_QUESTION = 'reader/op_decode_question'


def _parse_texts(tokens, offsets, lengths):
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


def get_input_fn(options):
  """Returns a function that generate input examples.

  Args:
    options: an instance of reader_pb2.Reader.

  Returns:
    input_fn: a callable that returns a dataset.
  """
  if not isinstance(options, reader_pb2.AdViSEReader):
    raise ValueError('options has to be an instance of Reader.')

  def _parse_fn(example):
    """Parses tf::Example proto.

    Args:
      example: a tf::Example proto.

    Returns:
      feature_dict: a dict mapping from names to tensors.
    """
    example_fmt = {
        TFExampleDataFields.image_id:
        tf.FixedLenFeature([], tf.int64),
        TFExampleDataFields.image_feature:
        tf.FixedLenFeature([options.feature_dimensions], tf.float32),
        TFExampleDataFields.roi_feature_size:
        tf.FixedLenFeature([], tf.int64),
        TFExampleDataFields.roi_feature:
        tf.VarLenFeature(tf.float32),
        TFExampleDataFields.slogan_text_string:
        tf.VarLenFeature(tf.string),
        TFExampleDataFields.slogan_text_offset:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.slogan_text_length:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.groundtruth_text_string:
        tf.VarLenFeature(tf.string),
        TFExampleDataFields.groundtruth_text_offset:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.groundtruth_text_length:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.question_text_string:
        tf.VarLenFeature(tf.string),
        TFExampleDataFields.question_text_offset:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.question_text_length:
        tf.VarLenFeature(tf.int64),
    }
    parsed = tf.parse_single_example(
        example, example_fmt, name=_OP_PARSE_SINGLE_EXAMPLE)

    feature_dict = {
        InputDataFields.image_id: parsed[TFExampleDataFields.image_id],
    }

    # Decode image feature.

    with tf.name_scope(_OP_DECODE_IMAGE):
      (feature_dict[InputDataFields.image_feature],
       feature_dict[InputDataFields.roi_feature_size]) = (
           parsed[TFExampleDataFields.image_feature],
           parsed[TFExampleDataFields.roi_feature_size])
      feature_dict[InputDataFields.roi_feature] = tf.reshape(
          tf.sparse_tensor_to_dense(parsed[InputDataFields.roi_feature]),
          [-1, options.feature_dimensions])

    tuples = [
        (_OP_DECODE_SLOGAN, TFExampleDataFields.slogan_text_string,
         TFExampleDataFields.slogan_text_offset,
         TFExampleDataFields.slogan_text_length,
         InputDataFields.slogan_text_size, InputDataFields.slogan_text_string,
         InputDataFields.slogan_text_length),
        (_OP_DECODE_GROUNDTRUTH, TFExampleDataFields.groundtruth_text_string,
         TFExampleDataFields.groundtruth_text_offset,
         TFExampleDataFields.groundtruth_text_length,
         InputDataFields.groundtruth_text_size,
         InputDataFields.groundtruth_text_string,
         InputDataFields.groundtruth_text_length),
        (_OP_DECODE_QUESTION, TFExampleDataFields.question_text_string,
         TFExampleDataFields.question_text_offset,
         TFExampleDataFields.question_text_length,
         InputDataFields.question_text_size,
         InputDataFields.question_text_string,
         InputDataFields.question_text_length),
    ]

    for (name_scope, input_string_field, input_offset_field, input_length_field,
         output_size_field, output_string_field, output_length_field) in tuples:
      with tf.name_scope(name_scope):
        (feature_dict[output_size_field], feature_dict[output_string_field],
         feature_dict[output_length_field]) = _parse_texts(
             tokens=tf.sparse_tensor_to_dense(
                 parsed[input_string_field], default_value=""),
             offsets=tf.sparse_tensor_to_dense(
                 parsed[input_offset_field], default_value=0),
             lengths=tf.sparse_tensor_to_dense(
                 parsed[input_length_field], default_value=0))

    return feature_dict

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
    dataset = dataset.map(
        map_func=_parse_fn, num_parallel_calls=options.map_num_parallel_calls)
    if options.is_training:
      dataset = dataset.cache()
      dataset = dataset.repeat().shuffle(options.shuffle_buffer_size)

    padded_shapes = {
        InputDataFields.image_id: [],
        InputDataFields.image_feature: [options.feature_dimensions],
        InputDataFields.roi_feature: [None, options.feature_dimensions],
        InputDataFields.roi_feature_size: [],
    }

    tuples = [
        (_OP_DECODE_SLOGAN, InputDataFields.slogan_text_size,
         InputDataFields.slogan_text_string,
         InputDataFields.slogan_text_length),
        (_OP_DECODE_GROUNDTRUTH, InputDataFields.groundtruth_text_size,
         InputDataFields.groundtruth_text_string,
         InputDataFields.groundtruth_text_length),
        (_OP_DECODE_QUESTION, InputDataFields.question_text_size,
         InputDataFields.question_text_string,
         InputDataFields.question_text_length),
    ]

    for name_scope, output_size_field, output_string_field, output_length_field in tuples:
      padded_shapes[output_size_field] = []
      padded_shapes[output_string_field] = [None, None]
      padded_shapes[output_length_field] = [None]

    dataset = dataset.padded_batch(
        options.batch_size, padded_shapes=padded_shapes, drop_remainder=True)
    dataset = dataset.prefetch(options.prefetch_buffer_size)
    return dataset

  return _input_fn
