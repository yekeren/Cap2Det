
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from protos import optimizer_pb2


def build_optimizer(options, learning_rate=0.1):
  """Builds optimizer from options.

  Args:
    options: an instance of optimizer_pb2.Optimizer.
    learning_rate: a scalar tensor denoting the learning rate.

  Returns:
    a tensorflow optimizer instance.

  Raises:
    ValueError: if options is invalid.
  """
  if not isinstance(options, optimizer_pb2.Optimizer):
    raise ValueError('The options has to be an instance of Optimizer.')

  optimizer = options.WhichOneof('optimizer')

  if 'sgd' == optimizer:
    options = options.sgd
    return tf.train.GradientDescentOptimizer(learning_rate,
        use_locking=options.use_locking)

  if 'adagrad' == optimizer:
    options = options.adagrad
    return tf.train.AdagradOptimizer(learning_rate,
        initial_accumulator_value=options.initial_accumulator_value,
        use_locking=options.use_locking)

  if 'adam' == optimizer:
    options = options.adam
    return tf.train.AdamOptimizer(learning_rate,
        beta1=options.beta1, beta2=options.beta2,
        epsilon=options.epsilon, use_locking=options.use_locking)

  if 'rmsprop' == optimizer:
    options = options.rmsprop
    return tf.train.RMSPropOptimizer(learning_rate,
        decay=options.decay, momentum=options.momentum,
        epsilon=options.epsilon, use_locking=options.use_locking,
        centered=options.centered)

  raise ValueError('Invalid optimizer: {}.'.format(optimizer))
