from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import tensorflow as tf
from core import imgproc
from core import plotlib

_TESTDATA = "testdata"
_TESTFILE = "54666.jpg"
_TMPDIR = "tmp"
_XIMGPROC_MODEL = "zoo/ximgproc/model.yml"

tf.logging.set_verbosity(tf.logging.INFO)


class ImgProcTest(tf.test.TestCase):

  def test_gaussian_kernel(self):

    # 3x3 kernel.

    kernel = imgproc._py_gaussian_kernel(3)
    self.assertNear(kernel.sum(), 1.0, err=1e-8)
    self.assertEqual(kernel.shape, (3, 3))
    self.assertEqual(kernel[1, 1], kernel.max())

    # 5x5 kernel.

    kernel = imgproc._py_gaussian_kernel(5)
    self.assertNear(kernel.sum(), 1.0, err=1e-8)
    self.assertEqual(kernel.shape, (5, 5))
    self.assertEqual(kernel[2, 2], kernel.max())

  def test_gaussian_filter(self):

    image = tf.placeholder(tf.float32, shape=[None, None, None, 1])
    outputs = imgproc.gaussian_filter(image, ksize=5)

    with self.test_session() as sess:

      # Print smoothed image.

      image_data = np.zeros((1, 10, 10, 1), dtype=np.float32)
      image_data[:, 3:7, 3:7, :] = 1.0

      image_smoothed = sess.run(
          outputs[0, :, :, 0], feed_dict={image: image_data})
      tf.logging.info("\n%s", image_smoothed)

  def test_calc_integral_image(self):
    tf.reset_default_graph()

    data = tf.placeholder(tf.float32, shape=[None, None, None])
    cumsum = imgproc.calc_integral_image(tf.expand_dims(data, axis=0))[0]

    with self.test_session() as sess:

      # 2x3 image, 1 channel.

      cumsum_value = sess.run(
          cumsum, feed_dict={data: [[[1], [2], [3]], [[4], [5], [6]]]})
      self.assertAllClose(
          cumsum_value,
          [[[0], [0], [0], [0]], [[0], [1], [3], [6]], [[0], [5], [12], [21]]])

      # 3x3 image, 1 channel.

      cumsum_value = sess.run(
          cumsum,
          feed_dict={data: [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]})
      self.assertAllClose(cumsum_value,
                          [[[0], [0], [0], [0]], [[0], [1], [3], [6]],
                           [[0], [5], [12], [21]], [[0], [12], [27], [45]]])

      # 3x3 image, 2 channels.

      cumsum_value = sess.run(
          cumsum,
          feed_dict={
              data: [[[1, 0], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]],
                     [[7, 7], [8, 8], [9, 9]]]
          })
      self.assertAllClose(
          cumsum_value,
          [[[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [1, 0], [3, 2], [6, 5]],
           [[0, 0], [5, 4], [12, 11], [21, 20]],
           [[0, 0], [12, 11], [27, 26], [45, 44]]])

  def test_calc_cumsum_2d(self):
    tf.reset_default_graph()

    image = tf.placeholder(tf.float32, shape=[None, None, None])
    box = tf.placeholder(tf.int64, shape=[None, 4])
    cumsum = imgproc.calc_cumsum_2d(
        tf.expand_dims(image, axis=0), tf.expand_dims(box, axis=0))[0]

    with self.test_session() as sess:

      # 3x3 image, 1 channel.

      result = sess.run(
          cumsum,
          feed_dict={
              image: [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]],
              box: [
                  [0, 0, 1, 1],
                  [2, 2, 3, 3],
                  [1, 1, 2, 2],
                  [0, 0, 2, 2],
                  [1, 1, 3, 3],
                  [0, 0, 3, 3],
                  [0, 0, 2, 3],
              ]
          })
      self.assertAllClose(result, [[1], [9], [5], [12], [28], [45], [21]])

      # 3x3 image, 2 channels.

      result = sess.run(
          cumsum,
          feed_dict={
              image: [[[1, 1], [2, 2], [3, 3]], [[4, 1], [5, 2], [6, 3]],
                      [[7, 1], [8, 2], [9, 3]]],
              box: [
                  [0, 0, 1, 1],
                  [2, 2, 3, 3],
                  [1, 1, 2, 2],
                  [0, 0, 2, 2],
                  [1, 1, 3, 3],
                  [0, 0, 3, 3],
                  [0, 0, 2, 3],
              ]
          })
      self.assertAllClose(
          result,
          [[1, 1], [9, 3], [5, 2], [12, 6], [28, 10], [45, 18], [21, 12]])

  def test_resize_image_to_size(self):
    tf.reset_default_graph()

    image = tf.placeholder(tf.float32, shape=[None, None, None])
    new_height = tf.placeholder(tf.int32, shape=[])
    new_width = tf.placeholder(tf.int32, shape=[])
    resized_image = imgproc.resize_image_to_size(
        image, new_height=new_height, new_width=new_width)

    with self.test_session() as sess:
      img, img_shape = sess.run(
          resized_image,
          feed_dict={
              image: np.zeros([300, 400, 3]),
              new_height: 600,
              new_width: 800,
          })
      self.assertEqual(img.shape, (600, 800, 3))
      self.assertAllEqual(img_shape, [600, 800, 3])

  def test_resize_image_to_max_dimension(self):
    tf.reset_default_graph()

    image = tf.placeholder(tf.float32, shape=[None, None, 3])

    # pad_to_max_dimension = False.

    resized_image = imgproc.resize_image_to_max_dimension(
        image, max_dimension=800, pad_to_max_dimension=False)
    with self.test_session() as sess:
      img, img_shape = sess.run(
          resized_image, feed_dict={
              image: np.zeros([300, 400, 3]),
          })
      self.assertEqual(img.shape, (600, 800, 3))
      self.assertAllEqual(img_shape, [600, 800, 3])

      img, img_shape = sess.run(
          resized_image, feed_dict={
              image: np.zeros([400, 300, 3]),
          })
      self.assertEqual(img.shape, (800, 600, 3))
      self.assertAllEqual(img_shape, [800, 600, 3])

    # pad_to_max_dimension = True.

    resized_image = imgproc.resize_image_to_max_dimension(
        image, max_dimension=800, pad_to_max_dimension=True)
    with self.test_session() as sess:
      img, img_shape = sess.run(
          resized_image, feed_dict={
              image: np.zeros([300, 400, 3]),
          })
      self.assertEqual(img.shape, (800, 800, 3))
      self.assertAllEqual(img_shape, [600, 800, 3])

  def test_resize_image_to_min_dimension(self):
    tf.reset_default_graph()

    image = tf.placeholder(tf.float32, shape=[None, None, 3])

    resized_image = imgproc.resize_image_to_min_dimension(
        image, min_dimension=900)
    with self.test_session() as sess:
      img, img_shape = sess.run(
          resized_image, feed_dict={
              image: np.zeros([300, 400, 3]),
          })
      self.assertEqual(img.shape, (900, 1200, 3))
      self.assertAllEqual(img_shape, [900, 1200, 3])

      img, img_shape = sess.run(
          resized_image, feed_dict={
              image: np.zeros([400, 300, 3]),
          })
      self.assertEqual(img.shape, (1200, 900, 3))
      self.assertAllEqual(img_shape, [1200, 900, 3])


#  def test_calc_box_saliency(self):
#    tf.reset_default_graph()
#
#    saliency_map = tf.placeholder(tf.float32, shape=[None, None])
#    box = tf.placeholder(tf.int64, shape=[None, 4])
#    border_ratio = tf.placeholder(tf.float32, shape=[])
#    alpha = tf.placeholder(tf.float32, shape=[])
#
#    box_saliency = imgproc.calc_box_saliency(
#        tf.expand_dims(saliency_map, axis=0),
#        tf.expand_dims(box, axis=0),
#        border_ratio, alpha)[0]
#
#    with self.test_session() as sess:
#      # border_ratio = 0.667, box_size = 3x3.
#      with self.assertRaises(tf.errors.InvalidArgumentError):
#        result = sess.run(box_saliency, feed_dict={
#            saliency_map: [[1, 2, 3], [4, 8, 6], [7, 8, 9]],
#            border_ratio: 2.0 / 3.0,
#            alpha: 1.0,
#            box: [
#            [0, 0, 3, 3],
#            ] })
#
#      # border_ratio = 0, box_size = 3x3.
#      result = sess.run(box_saliency, feed_dict={
#          saliency_map: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#          border_ratio: 0,
#          alpha: 1.0,
#          box: [
#          [0, 0, 3, 3],
#          ] })
#      self.assertAllClose(result, [5.0])
#
#      # border_ratio = 0.333, box_size = 3x3.
#      result = sess.run(box_saliency, feed_dict={
#          saliency_map: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#          border_ratio: 1.0 / 3.0,
#          alpha: 1.0,
#          box: [
#          [0, 0, 3, 3],
#          ] })
#      self.assertAllClose(result, [0.0])
#
#      # border_ratio = 0.333, box_size = 3x3.
#      result = sess.run(box_saliency, feed_dict={
#          saliency_map: [[1, 2, 3], [4, 8, 6], [7, 8, 9]],
#          border_ratio: 1.0 / 3.0,
#          alpha: 1.0,
#          box: [
#          [0, 0, 3, 3],
#          ] })
#      self.assertAllClose(result, [3.0])
#
#      # border_ratio = 0.333, alpha=0.5, box_size = 3x3.
#      result = sess.run(box_saliency, feed_dict={
#          saliency_map: [[1, 2, 3], [4, 8, 6], [7, 8, 9]],
#          border_ratio: 1.0 / 3.0,
#          alpha: 0.5,
#          box: [
#          [0, 0, 3, 3],
#          ] })
#      self.assertAllClose(result, [5.5])

#  def test_py_get_edge_boxes(self):
#    filename = os.path.join(_TESTDATA, _TESTFILE)
#    image = cv2.imread(filename)[:, :, ::-1].copy()
#
#    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(_XIMGPROC_MODEL)
#    edge_boxes = cv2.ximgproc.createEdgeBoxes()
#
#    num_boxes, boxes = imgproc._py_get_edge_boxes(
#        np.float32(image), edge_detection, edge_boxes, max_num_boxes=5)
#    self.assertEqual(5, num_boxes)
#    self.assertAllEqual(boxes.shape, [5, 4])
#    image_with_boxes = plotlib._py_draw_rectangles(
#        image, boxes, color=[0, 0, 255], thickness=4)
#
#    filename = os.path.join(_TMPDIR, "py_edge_boxes.jpg")
#    cv2.imwrite(filename, image_with_boxes[:, :, ::-1])
#    tf.logging.info("The image with edge boxes is written to %s.", filename)
#
#  def test_get_edge_boxes(self):
#    images = tf.placeholder(tf.uint8, shape=[None, None, None, 3])
#    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(_XIMGPROC_MODEL)
#    edge_boxes = cv2.ximgproc.createEdgeBoxes()
#
#    num_boxes, boxes = imgproc.get_edge_boxes(
#        tf.cast(images, tf.float32),
#        edge_detection,
#        edge_boxes,
#        max_num_boxes=5)
#
#    with self.test_session() as sess:
#
#      # Empty image.
#
#      black_im = np.zeros((224, 224, 3), np.float32)
#      num_boxes_val, boxes_val = sess.run([num_boxes[0], boxes[0]],
#                                          feed_dict={images: [black_im]})
#      self.assertEqual(0, num_boxes_val)
#      self.assertAllEqual(boxes_val.shape, [5, 4])
#
#      # Normal image.
#
#      filename = os.path.join(_TESTDATA, _TESTFILE)
#      rgb_im = cv2.imread(filename)[:, :, ::-1].copy()
#      num_boxes_val, boxes_val = sess.run([num_boxes[0], boxes[0]],
#                                          feed_dict={images: [rgb_im]})
#      self.assertEqual(5, num_boxes_val)
#      self.assertAllEqual(boxes_val.shape, [5, 4])
#
#    image_with_boxes = plotlib._py_draw_rectangles(
#        rgb_im, boxes_val, color=[0, 255, 0], thickness=4)
#
#    filename = os.path.join(_TMPDIR, "edge_boxes.jpg")
#    cv2.imwrite(filename, image_with_boxes[:, :, ::-1])
#    tf.logging.info("The image with edge boxes is written to %s.", filename)

if __name__ == '__main__':
  tf.test.main()
