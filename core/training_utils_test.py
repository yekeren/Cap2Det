from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from google.protobuf import text_format

from core import training_utils
from protos import optimizer_pb2


class TrainingUtilsTest(tf.test.TestCase):

  def test_build_optimizer(self):
    # Gradient descent optimizer.

    options_str = r"""
      sgd {
      }
    """
    options = optimizer_pb2.Optimizer()
    text_format.Merge(options_str, options)
    opt = training_utils.build_optimizer(options)
    self.assertIsInstance(opt, tf.train.GradientDescentOptimizer)

    # Adagrad optimizer.

    options_str = r"""
      adagrad {
      }
    """
    options = optimizer_pb2.Optimizer()
    text_format.Merge(options_str, options)
    opt = training_utils.build_optimizer(options)
    self.assertIsInstance(opt, tf.train.AdagradOptimizer)

    # Adam optimizer.

    options_str = r"""
      adam {
      }
    """
    options = optimizer_pb2.Optimizer()
    text_format.Merge(options_str, options)
    opt = training_utils.build_optimizer(options)
    self.assertIsInstance(opt, tf.train.AdamOptimizer)

    # Rmsprop optimizer.

    options_str = r"""
      rmsprop {
      }
    """
    options = optimizer_pb2.Optimizer()
    text_format.Merge(options_str, options)
    opt = training_utils.build_optimizer(options)
    self.assertIsInstance(opt, tf.train.RMSPropOptimizer)


if __name__ == '__main__':
  tf.test.main()
