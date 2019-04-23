from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from core.standard_fields import InputDataFields
from core.standard_fields import TFExampleDataFields
from protos import reader_pb2


class TFOperations(object):
  """Names of important operations."""
  parse_single_example = 'parse_single_example'
  decode_feature = 'decode_feature'

  decode_label = 'decode_label'
  decode_question = 'decode_question'
  decode_ocr = 'decode_ocr'
  decode_densecap = 'decode_densecap'

  decode_full_ocr = 'decode_full_ocr'
  decode_full_densecap = 'decode_full_densecap'

  get_groundtruth_mask = 'get_groundtruth_mask'
  mine_negative_examples = 'mine_negative_examples'
  retrieve_in_batch_examples = 'retrieve_in_batch_examples'


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
        TFExampleDataFields.image_id:
        tf.FixedLenFeature([], tf.int64),
        TFExampleDataFields.image_feature:
        tf.FixedLenFeature([options.feature_dimensions], tf.float32),
        TFExampleDataFields.roi_feature_size:
        tf.FixedLenFeature([], tf.int64),
        TFExampleDataFields.roi_feature:
        tf.VarLenFeature(tf.float32),
        TFExampleDataFields.symbol_feature:
        tf.FixedLenFeature([options.symbol_dimensions], tf.float32),
        TFExampleDataFields.label_text_string:
        tf.VarLenFeature(tf.string),
        TFExampleDataFields.label_text_offset:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.label_text_length:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.question_text_string:
        tf.VarLenFeature(tf.string),
        TFExampleDataFields.question_text_offset:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.question_text_length:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.densecap_text_string:
        tf.VarLenFeature(tf.string),
        TFExampleDataFields.densecap_text_offset:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.densecap_text_length:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.ocr_text_string:
        tf.VarLenFeature(tf.string),
        TFExampleDataFields.ocr_text_offset:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.ocr_text_length:
        tf.VarLenFeature(tf.int64),
    }
    parsed = tf.parse_single_example(
        example, example_fmt, name=TFOperations.parse_single_example)

    feature_dict = {
        InputDataFields.image_id: parsed[TFExampleDataFields.image_id],
        InputDataFields.symbol_feature:
        parsed[TFExampleDataFields.symbol_feature],
    }

    # Decode image feature.

    with tf.name_scope(TFOperations.decode_feature):
      (feature_dict[InputDataFields.image_feature],
       feature_dict[InputDataFields.roi_feature_size]) = (
           parsed[TFExampleDataFields.image_feature],
           parsed[TFExampleDataFields.roi_feature_size])
      feature_dict[InputDataFields.roi_feature] = tf.reshape(
          tf.sparse_tensor_to_dense(parsed[InputDataFields.roi_feature]),
          [-1, options.feature_dimensions])

    valid_ops = set([op for op in options.decode_annotation])

    # Decode texts.

    tuples = [
        (TFOperations.decode_label, TFExampleDataFields.label_text_string,
         TFExampleDataFields.label_text_offset,
         TFExampleDataFields.label_text_length, InputDataFields.label_text_size,
         InputDataFields.label_text_string, InputDataFields.label_text_length),
        (TFOperations.decode_question, TFExampleDataFields.question_text_string,
         TFExampleDataFields.question_text_offset,
         TFExampleDataFields.question_text_length,
         InputDataFields.question_text_size,
         InputDataFields.question_text_string,
         InputDataFields.question_text_length),
        (TFOperations.decode_densecap, TFExampleDataFields.densecap_text_string,
         TFExampleDataFields.densecap_text_offset,
         TFExampleDataFields.densecap_text_length,
         InputDataFields.densecap_text_size,
         InputDataFields.densecap_text_string,
         InputDataFields.densecap_text_length),
        (TFOperations.decode_ocr, TFExampleDataFields.ocr_text_string,
         TFExampleDataFields.ocr_text_offset,
         TFExampleDataFields.ocr_text_length, InputDataFields.ocr_text_size,
         InputDataFields.ocr_text_string, InputDataFields.ocr_text_length),
    ]

    for (name_scope, input_string_field, input_offset_field, input_length_field,
         output_size_field, output_string_field, output_length_field) in tuples:
      if not name_scope in valid_ops:
        tf.logging.info('Skip decoding %s', name_scope)
        continue
      with tf.name_scope(name_scope):
        (feature_dict[output_size_field], feature_dict[output_string_field],
         feature_dict[output_length_field]) = _parse_texts(
             tokens=tf.sparse_tensor_to_dense(
                 parsed[input_string_field], default_value=""),
             offsets=tf.sparse_tensor_to_dense(
                 parsed[input_offset_field], default_value=0),
             lengths=tf.sparse_tensor_to_dense(
                 parsed[input_length_field], default_value=0))

    # Decode full texts.

    tuples = [
        (TFOperations.decode_full_densecap,
         TFExampleDataFields.densecap_text_string,
         TFExampleDataFields.densecap_text_offset,
         TFExampleDataFields.densecap_text_length,
         InputDataFields.densecap_full_text_string,
         InputDataFields.densecap_full_text_length),
        (TFOperations.decode_full_ocr, TFExampleDataFields.ocr_text_string,
         TFExampleDataFields.ocr_text_offset,
         TFExampleDataFields.ocr_text_length,
         InputDataFields.ocr_full_text_string,
         InputDataFields.ocr_full_text_length),
    ]

    for (name_scope, input_string_field, input_offset_field, input_length_field,
         output_full_string_field, output_full_length_field) in tuples:
      if not name_scope in valid_ops:
        tf.logging.info('Skip decoding %s', name_scope)
        continue
      with tf.name_scope(name_scope):
        feature_dict[output_full_string_field] = tf.sparse_tensor_to_dense(
            parsed[input_string_field], default_value="")
        feature_dict[output_full_length_field] = tf.reduce_sum(
            tf.sparse_tensor_to_dense(
                parsed[input_length_field], default_value=0))
        if options.HasField('max_full_text_length'):
          max_full_text_length = options.max_full_text_length
          feature_dict[output_full_string_field] = feature_dict[
              output_full_string_field][:max_full_text_length]
          feature_dict[output_full_length_field] = tf.minimum(
              feature_dict[output_full_length_field], max_full_text_length)

    return feature_dict

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

    padded_shapes = {
        InputDataFields.image_id: [],
        InputDataFields.image_feature: [options.feature_dimensions],
        InputDataFields.roi_feature: [None, options.feature_dimensions],
        InputDataFields.roi_feature_size: [],
        InputDataFields.symbol_feature: [options.symbol_dimensions],
    }
    valid_ops = set([op for op in options.decode_annotation])

    # Texts.
    tuples = [
        (TFOperations.decode_label, InputDataFields.label_text_size,
         InputDataFields.label_text_string, InputDataFields.label_text_length),
        (TFOperations.decode_question, InputDataFields.question_text_size,
         InputDataFields.question_text_string,
         InputDataFields.question_text_length),
        (TFOperations.decode_densecap, InputDataFields.densecap_text_size,
         InputDataFields.densecap_text_string,
         InputDataFields.densecap_text_length),
        (TFOperations.decode_ocr, InputDataFields.ocr_text_size,
         InputDataFields.ocr_text_string, InputDataFields.ocr_text_length),
    ]

    for name_scope, output_size_field, output_string_field, output_length_field in tuples:
      if not name_scope in valid_ops:
        continue
      padded_shapes[output_size_field] = []
      padded_shapes[output_string_field] = [None, None]
      padded_shapes[output_length_field] = [None]

    # Full texts.
    tuples = [
        (TFOperations.decode_full_densecap,
         InputDataFields.densecap_full_text_string,
         InputDataFields.densecap_full_text_length),
        (TFOperations.decode_full_ocr, InputDataFields.ocr_full_text_string,
         InputDataFields.ocr_full_text_length),
    ]

    for name_scope, output_full_string_field, output_full_length_field in tuples:
      if not name_scope in valid_ops:
        continue
      padded_shapes[output_full_string_field] = [None]
      padded_shapes[output_full_length_field] = []

    dataset = dataset.padded_batch(
        options.batch_size, padded_shapes=padded_shapes, drop_remainder=True)
    dataset = dataset.prefetch(options.prefetch_buffer_size)
    return dataset

  return _input_fn
