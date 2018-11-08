from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from core import utils


class UtilsTest(tf.test.TestCase):

  def test_masked_maximum(self):
    tf.reset_default_graph()

    data = tf.placeholder(tf.float32, shape=[None, None])
    mask = tf.placeholder(tf.float32, shape=[None, None])
    masked_maximums = utils.masked_maximum(data, mask)

    with self.test_session() as sess:
      result = sess.run(
          masked_maximums,
          feed_dict={
              data: [[-2.0, 1.0, 2.0, -1.0, 0.0],
                     [-2.0, -1.0, -3.0, -5.0, -4.0]],
              mask: [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
          })
      self.assertAllClose(result, [[2.0], [-1.0]])

      result = sess.run(
          masked_maximums,
          feed_dict={
              data: [[-2.0, 1.0, 2.0, -1.0, 0.0],
                     [-2.0, -1.0, -3.0, -5.0, -4.0]],
              mask: [[1, 1, 0, 1, 1], [0, 0, 1, 1, 1]]
          })
      self.assertAllClose(result, [[1.0], [-3.0]])

      result = sess.run(
          masked_maximums,
          feed_dict={
              data: [[-2.0, 1.0, 2.0, -1.0, 0.0],
                     [-2.0, -1.0, -3.0, -5.0, -4.0]],
              mask: [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
          })
      self.assertAllClose(result, [[-2.0], [-5.0]])

  def test_masked_minimum(self):
    tf.reset_default_graph()

    data = tf.placeholder(tf.float32, shape=[None, None])
    mask = tf.placeholder(tf.float32, shape=[None, None])
    masked_minimums = utils.masked_minimum(data, mask)

    with self.test_session() as sess:
      result = sess.run(
          masked_minimums,
          feed_dict={
              data: [[-2.0, 1.0, 2.0, -1.0, 0.0],
                     [-2.0, -1.0, -3.0, -5.0, -4.0]],
              mask: [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
          })
      self.assertAllClose(result, [[-2.0], [-5.0]])

      result = sess.run(
          masked_minimums,
          feed_dict={
              data: [[-2.0, 1.0, 2.0, -1.0, 0.0],
                     [-2.0, -1.0, -3.0, -5.0, -4.0]],
              mask: [[0, 1, 1, 0, 1], [1, 1, 1, 0, 1]]
          })
      self.assertAllClose(result, [[0.0], [-4.0]])

      result = sess.run(
          masked_minimums,
          feed_dict={
              data: [[-2.0, 1.0, 2.0, -1.0, 0.0],
                     [-2.0, -1.0, -3.0, -5.0, -4.0]],
              mask: [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
          })
      self.assertAllClose(result, [[2.0], [-1.0]])

  def test_masked_softmax(self):
    tf.reset_default_graph()

    data = tf.placeholder(tf.float32, shape=[None, None])
    mask = tf.placeholder(tf.float32, shape=[None, None])
    masked_softmax = utils.masked_softmax(data, mask)

    with self.test_session() as sess:
      result = sess.run(
          masked_softmax,
          feed_dict={
              data: [[1, 1, 1, 1], [1, 1, 1, 1]],
              mask: [[1, 1, 1, 1], [1, 1, 1, 1]]
          })
      self.assertAllClose(result,
                          [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])

      result = sess.run(
          masked_softmax,
          feed_dict={
              data: [[1, 1, 1, 1], [1, 1, 1, 1]],
              mask: [[1, 1, 0, 0], [0, 0, 1, 1]]
          })
      self.assertAllClose(result, [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])


if __name__ == '__main__':
  tf.test.main()
