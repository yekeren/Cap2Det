from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import box_utils


class BoxUtilsTest(tf.test.TestCase):

  def testScaleToNewSize(self):
    """Test scale_to_new_size."""
    tf.reset_default_graph()
    box = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    img_shape = tf.placeholder(dtype=tf.int32, shape=[2])
    pad_shape = tf.placeholder(dtype=tf.int32, shape=[2])
    box_scaled = box_utils.scale_to_new_size(box, img_shape, pad_shape)

    with self.test_session() as sess:
      box_scaled = sess.run(
          box_scaled,
          feed_dict={
              box: [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 1.0],
                    [0.0, 0.0, 0.5, 0.5]],
              img_shape: [1, 1],
              pad_shape: [2, 1],
          })
      self.assertAllClose(
          box_scaled,
          [[0.0, 0.0, 0.5, 1.0], [0.0, 0.0, 0.25, 1.0], [0.0, 0.0, 0.25, 0.5]])

  def testFlipLeftRight(self):
    """Test flip_left_right."""
    tf.reset_default_graph()
    box = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    box_flipped = box_utils.flip_left_right(box)

    with self.test_session() as sess:
      box_flipped = sess.run(
          box_flipped,
          feed_dict={
              box: [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 1.0],
                    [0.0, 0.0, 0.5, 0.5]]
          })
      self.assertAllClose(
          box_flipped,
          [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 1.0], [0.0, 0.5, 0.5, 1.0]])

  def testArea(self):
    """Test area."""
    tf.reset_default_graph()
    box = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    area = box_utils.area(box)

    with self.test_session() as sess:
      area = sess.run(
          area,
          feed_dict={
              box: [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 1.0],
                    [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, -1.0, -1.0],
                    [0.0, 0.0, 0.0, 0.0]]
          })
      self.assertAllClose(area, [1.0, 0.5, 0.25, 0.0, 0.0])

  def testIntersect(self):
    """ Test intersect. """
    tf.reset_default_graph()
    box1 = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    box2 = tf.placeholder(dtype=tf.float32, shape=[None, 4])

    intersect = box_utils.intersect(box1, box2)

    with self.test_session() as sess:
      intersect = sess.run(
          intersect,
          feed_dict={
              box1: [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 3.0],
                     [0.0, 0.0, 3.0, 2.0], [0.0, 0.0, 1.0, 1.0],
                     [0.0, 0.0, 1.0, 1.0]],
              box2: [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0],
                     [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0],
                     [2.0, 2.0, 1.0, 1.0]]
          })
      self.assertAllClose(
          intersect,
          [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0],
           [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0]])

  def testIoU(self):
    """Test iou. """
    tf.reset_default_graph()
    box1 = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    box2 = tf.placeholder(dtype=tf.float32, shape=[None, 4])

    iou = box_utils.iou(box1, box2)
    with self.test_session() as sess:
      iou = sess.run(
          iou,
          feed_dict={
              box1: [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 3.0],
                     [1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0],
                     [0.0, 0.0, 1.0, 1.0]],
              box2: [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0],
                     [0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0],
                     [2.0, 2.0, 1.0, 1.0]]
          })
      self.assertAllClose(iou, [0.25, 1.0 / 6, 1.0 / 6, 0.0, 0.0])


if __name__ == "__main__":
  tf.test.main()
