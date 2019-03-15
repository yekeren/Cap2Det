from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
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

  def test_masked_sum(self):
    tf.reset_default_graph()

    data = tf.placeholder(tf.float32, shape=[None, None])
    mask = tf.placeholder(tf.float32, shape=[None, None])
    masked_sums = utils.masked_sum(data, mask)

    with self.test_session() as sess:
      result = sess.run(
          masked_sums,
          feed_dict={
              data: [[1, 2, 3], [4, 5, 6]],
              mask: [[1, 0, 1], [0, 1, 0]]
          })
      self.assertAllClose(result, [[4], [5]])

      result = sess.run(
          masked_sums,
          feed_dict={
              data: [[1, 2, 3], [4, 5, 6]],
              mask: [[0, 1, 0], [1, 0, 1]]
          })
      self.assertAllClose(result, [[2], [10]])

  def test_masked_avg(self):
    tf.reset_default_graph()

    data = tf.placeholder(tf.float32, shape=[None, None])
    mask = tf.placeholder(tf.float32, shape=[None, None])
    masked_avgs = utils.masked_avg(data, mask)

    with self.test_session() as sess:
      result = sess.run(
          masked_avgs,
          feed_dict={
              data: [[1, 2, 3], [4, 5, 6]],
              mask: [[1, 0, 1], [0, 1, 0]]
          })
      self.assertAllClose(result, [[2], [5]])

      result = sess.run(
          masked_avgs,
          feed_dict={
              data: [[1, 2, 3], [4, 5, 6]],
              mask: [[0, 1, 0], [1, 0, 1]]
          })
      self.assertAllClose(result, [[2], [5]])

      result = sess.run(
          masked_avgs,
          feed_dict={
              data: [[1, 2, 3], [4, 5, 6]],
              mask: [[0, 0, 0], [0, 0, 0]]
          })
      self.assertAllClose(result, [[0], [0]])

  def test_masked_sum_nd(self):
    tf.reset_default_graph()

    data = tf.placeholder(tf.float32, shape=[None, None, None])
    mask = tf.placeholder(tf.float32, shape=[None, None])
    masked_sums = utils.masked_sum_nd(data, mask)

    with self.test_session() as sess:
      result = sess.run(
          masked_sums,
          feed_dict={
              data: [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]],
              mask: [[1, 0, 1], [0, 1, 0]]
          })
      self.assertAllClose(result, [[[6, 8]], [[9, 10]]])

  def test_masked_avg_nd(self):
    tf.reset_default_graph()

    data = tf.placeholder(tf.float32, shape=[None, None, None])
    mask = tf.placeholder(tf.float32, shape=[None, None])
    masked_avgs = utils.masked_avg_nd(data, mask)

    with self.test_session() as sess:
      result = sess.run(
          masked_avgs,
          feed_dict={
              data: [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]],
              mask: [[1, 0, 1], [0, 1, 0]]
          })
      self.assertAllClose(result, [[[3, 4]], [[9, 10]]])

      result = sess.run(
          masked_avgs,
          feed_dict={
              data: [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]],
              mask: [[0, 0, 0], [0, 0, 0]]
          })
      self.assertAllClose(result, [[[0, 0]], [[0, 0]]])

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

  def test_covariance(self):
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[None, None])
    cov = utils.covariance(x)

    data = np.array([[1, 4, 2], [5, 6, 24], [15, 1, 5], [7, 3, 8], [9, 4, 7]],
                    dtype=np.float32)
    with self.test_session() as sess:
      cov = sess.run(cov, feed_dict={x: data})

    self.assertAllClose(cov, np.cov(data, bias=True))


if __name__ == '__main__':
  tf.test.main()
