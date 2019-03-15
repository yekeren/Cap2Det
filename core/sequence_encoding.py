from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from core import utils


def _average_encoding(sequence_feature, sequence_length):
  """Encodes sequence using Average pooling.

  Args:
    sequence_feature: a [batch_sequence, max_sequence_length, feature_dimensions].
      float tensor.
    sequence_length: a [batch_sequence] int tensor.

  Returns:
    sequence_emb: A [batch_sequence, common_dimensions] float tensor, 
      representing the embedding vectors.
  """
  (_, max_sequence_length, _) = utils.get_tensor_shape(sequence_feature)

  mask = tf.sequence_mask(
      sequence_length, maxlen=max_sequence_length, dtype=tf.float32)

  sequence_emb = utils.masked_avg_nd(sequence_feature, mask, dim=1)
  sequence_emb = tf.squeeze(sequence_emb, axis=1)
  return sequence_emb


def _lstm_encoding(sequence_feature,
                   sequence_length,
                   number_of_layers=1,
                   hidden_units=200,
                   parallel_iterations=32,
                   is_training=False):
  """Encodes sequence using LSTM.

  Args:
    sequence_feature: A [batch_sequence, max_sequence_length, feature_dimensions] 
      float tensor representing the sequence feature.
    sequence_length: A [batch_sequence] int tensor representing the sequence 
      lengths.
    number_of_layers: Number of lstm layers.
    hidden_units: Number of lstm hidden units.
    is_training: if True, training graph is built.

  Returns:
    sequence_emb: A [batch_sequence, output_dimensions] float tensor, 
      representing the embedding vectors.
  """

  def lstm_cell():
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_units, forget_bias=1.0)
    return cell

  rnn_cell = tf.contrib.rnn.MultiRNNCell(
      [lstm_cell() for _ in range(number_of_layers)])

  outputs, state = tf.nn.dynamic_rnn(
      cell=rnn_cell,
      inputs=sequence_feature,
      sequence_length=sequence_length,
      parallel_iterations=parallel_iterations,
      dtype=tf.float32)

  return state[-1].h


def get_encode_fn(options):
  """Builds sequence encoding function based on the options.

  Args:
    options: An instance of sequence_encoding_pb2.SequenceEncoding.

  Returns:
    A function that takes sequence_feature, sequence_length and is_training 
      as parameter.
  """
  sequence_encoding_oneof = options.WhichOneof('sequence_encoding_oneof')

  if sequence_encoding_oneof == 'average_encoding':

    # Average Encoding.

    def _average_encoding_fn(sequence_feature,
                             sequence_length,
                             is_training=False):
      return _average_encoding(sequence_feature, sequence_length)

    return _average_encoding_fn

  if sequence_encoding_oneof == 'lstm_encoding':

    # LSTM Encoding.

    options = options.lstm_encoding

    def _lstm_encoding_fn(sequence_feature, sequence_length, is_training=False):
      return _lstm_encoding(
          sequence_feature,
          sequence_length,
          number_of_layers=options.number_of_layers,
          hidden_units=options.hidden_units,
          parallel_iterations=options.parallel_iterations,
          is_training=is_training)

    return _lstm_encoding_fn

  raise ValueError(
      'Unknown sequence encoding function: {}'.format(sequence_encoding_oneof))
