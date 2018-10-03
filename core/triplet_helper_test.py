
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from core import triplet_helper

tf.logging.set_verbosity(tf.logging.INFO)


class TripletHelperTest(tf.test.TestCase):

  def test_sample_random_negative_examples(self):
    g = tf.Graph()
    with g.as_default():
      batch_size = tf.placeholder(shape=[], dtype=tf.int64)
      pos_indices, neg_indices = triplet_helper.sample_random_negative_examples(
          batch_size, negatives_per_sample=4)

    with self.test_session(graph=g) as sess:
      pos, neg = sess.run([pos_indices, neg_indices], feed_dict={batch_size: 4})
      self.assertEqual(pos.shape, (16,))
      self.assertEqual(neg.shape, (16,))
      for i in range(16):
        self.assertNotEqual(pos[i], neg[i])

  def test_sample_all_negative_examples(self):
    g = tf.Graph()
    with g.as_default():
      batch_size = tf.placeholder(shape=[], dtype=tf.int64)
      (pos_indices, neg_indices
       ) = triplet_helper.sample_all_negative_examples(batch_size)

    with self.test_session(graph=g) as sess:
      pos, neg = sess.run([pos_indices, neg_indices], feed_dict={batch_size: 4})
      self.assertAllEqual(pos, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
      self.assertAllEqual(neg, np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]))

      pos, neg = sess.run([pos_indices, neg_indices], feed_dict={batch_size: 5})
      self.assertAllEqual(pos, np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3,
            3, 3, 3, 4, 4, 4, 4]))
      self.assertAllEqual(neg, np.array([1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0,
            1, 2, 4, 0, 1, 2, 3]))

  def test_triplet_semihard(self):
    g = tf.Graph()
    with g.as_default():
      distance_pos = tf.placeholder(shape=[None, None], dtype=tf.float32)
      distance_neg = tf.placeholder(shape=[None, None], dtype=tf.float32)
      num_captions_pos = tf.placeholder(shape=[None], dtype=tf.int32)
      num_captions_neg = tf.placeholder(shape=[None], dtype=tf.int32)

      distance_ap, distance_an = triplet_helper.triplet_semihard(
          distance_pos, distance_neg, num_captions_pos, num_captions_neg)

    # batch=3, max_num_captions=4.

    with self.test_session(graph=g) as sess:

      # Basics.

      (distance_ap_value, distance_an_value
       ) = sess.run([distance_ap, distance_an], feed_dict={ 
            distance_pos: [
              [1.0, 2.0, 3.0, 4.0],
              [4.0, 1.0, 2.0, 3.0],
              [3.0, 2.0, 1.0, 4.0],
            ],
            distance_neg: [
              [1.0, 2.0, 3.0, 4.0],
              [4.0, 1.0, 2.0, 3.0],
              [3.0, 2.0, 1.0, 4.0],
            ],
            num_captions_pos: [4, 4, 4],
            num_captions_neg: [4, 4, 4],
         })
      self.assertAllClose(distance_ap_value, [4.0, 4.0, 4.0])
      self.assertAllClose(distance_an_value, [1.0, 1.0, 1.0])

      # Test with padding.

      (distance_ap_value, distance_an_value
       ) = sess.run([distance_ap, distance_an], feed_dict={ 
            distance_pos: [
              [1.0, 2.0, 3.0, 4.0],
              [4.0, 1.0, 2.0, 3.0],
              [3.0, 2.0, 1.0, 4.0],
            ],
            distance_neg: [
              [3.0, 2.0, 1.0, 4.0],
              [4.0, 1.0, 2.0, 3.0],
              [3.0, 2.0, 1.0, 4.0],
            ],
            num_captions_pos: [4, 3, 2],
            num_captions_neg: [2, 3, 4],
         })
      self.assertAllClose(distance_ap_value, [4.0, 4.0, 3.0])
      self.assertAllClose(distance_an_value, [2.0, 1.0, 1.0])

      # Test with padding.

      (distance_ap_value, distance_an_value
       ) = sess.run([distance_ap, distance_an], feed_dict={ 
            distance_pos: [
              [1.0, 2.0, 3.0, 4.0],
              [4.0, 1.0, 2.0, 3.0],
              [3.0, 2.0, 1.0, 4.0],
            ],
            distance_neg: [
              [3.0, 2.0, 1.0, 4.0],
              [4.0, 1.0, 2.0, 3.0],
              [3.0, 2.0, 1.0, 4.0],
            ],
            num_captions_pos: [2, 2, 2],
            num_captions_neg: [1, 1, 1],
         })
      self.assertAllClose(distance_ap_value, [2.0, 4.0, 3.0])
      self.assertAllClose(distance_an_value, [3.0, 4.0, 3.0])

  def test_triplet_semihard_loss(self):
    g = tf.Graph()
    with g.as_default():
      batch = 4
      max_num_captions = 8

      distance_pos = tf.get_variable(
          "distance_pos", shape=[batch, max_num_captions])
      distance_neg = tf.get_variable(
          "distance_neg", shape=[batch, max_num_captions])
      num_captions_pos = tf.constant(
          max_num_captions, dtype=tf.int64, shape=[batch])
      num_captions_neg = tf.constant(
          max_num_captions, dtype=tf.int64, shape=[batch])

      loss = triplet_helper.triplet_semihard_loss(
          tf.identity(distance_pos), 
          tf.identity(distance_neg), 
          num_captions_pos, 
          num_captions_neg,
          margin=1.0)

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
      train_op = optimizer.minimize(loss)

    with self.test_session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())

      for i in range(500):
        loss_value, _ = sess.run([loss, train_op])
        if i % 100 == 0:
          tf.logging.info("Iteration %i, loss=%.3lf", i, loss_value)
      self.assertNear(loss_value, 0.0, err=1e-8)


if __name__ == '__main__':
  tf.test.main()
