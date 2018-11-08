from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from core import plotlib
from core import imgproc

_TMPDIR = "tmp"

tf.logging.set_verbosity(tf.logging.INFO)


class PlotLibTest(tf.test.TestCase):

  def test_py_convert_to_heatmap(self):
    kernel = imgproc._py_gaussian_kernel(ksize=100)
    kernel_heatmap = plotlib._py_convert_to_heatmap(kernel, normalize=True)
    kernel_heatmap = (kernel_heatmap * 255).astype(np.uint8)

    filename = _TMPDIR + "/py_convert_to_heatmap.png"
    cv2.imwrite(filename, kernel_heatmap[:, :, ::-1])  # RGB to BGR.
    tf.logging.info("The kernel image is written to %s.", filename)

  def test_convert_to_heatmap(self):
    image = tf.placeholder(tf.float32, shape=[None, None])
    heatmap = plotlib.convert_to_heatmap(
        tf.expand_dims(image, 0), normalize=True)

    with self.test_session() as sess:
      kernel = imgproc._py_gaussian_kernel(ksize=100)
      kernel_heatmap = sess.run(heatmap[0], feed_dict={image: kernel})
    kernel_heatmap = (kernel_heatmap * 255).astype(np.uint8)

    filename = _TMPDIR + "/convert_to_heatmap.png"
    cv2.imwrite(filename, kernel_heatmap[:, :, ::-1])  # RGB to BGR.
    tf.logging.info("The kernel image is written to %s.", filename)

  def test_py_draw_rectangles(self):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas = plotlib._py_draw_rectangles(
        image,
        boxes=[[0.1, 0.25, 0.9, 0.75], [0.2, 0.35, 0.8, 0.65]],
        scores=[0.1, 0.2],
        labels=["box1", "box2"],
        color=(255, 0, 0),
        thickness=1,
        fontscale=1.0)

    filename = _TMPDIR + "/py_draw_rectangles.png"
    cv2.imwrite(filename, canvas[:, :, ::-1])  # RGB to BGR.
    tf.logging.info("The image with rectangles is written to %s.", filename)

  def test_py_draw_caption(self):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas = plotlib._py_draw_caption(
        image,
        caption="hello, world",
        org=(10, 10),
        color=(255, 0, 0),
        thickness=1,
        fontscale=1.0)

    filename = _TMPDIR + "/py_draw_caption.png"
    cv2.imwrite(filename, canvas[:, :, ::-1])  # RGB to BGR.
    tf.logging.info("The image with caption is written to %s.", filename)

  def test_draw_rectangles(self):
    image = tf.placeholder(tf.uint8, shape=[None, None, 3])
    boxes = tf.placeholder(tf.float32, shape=[None, 4])
    scores = tf.placeholder(tf.float32, shape=[None])
    labels = tf.placeholder(tf.string, shape=[None])

    canvas = plotlib.draw_rectangles(
        image=tf.expand_dims(image, axis=0),
        boxes=tf.expand_dims(boxes, axis=0),
        scores=tf.expand_dims(scores, axis=0),
        labels=tf.expand_dims(labels, axis=0),
        color=(0, 0, 255),
        thickness=1,
        fontscale=1.0)

    with self.test_session() as sess:
      canvas = sess.run(
          canvas[0],
          feed_dict={
              image: np.zeros((480, 640, 3), dtype=np.uint8),
              boxes: [[0.1, 0.25, 0.9, 0.75], [0.2, 0.35, 0.8, 0.65]],
              scores: [0.123, 0.456],
              labels: ['bbox1', 'bbox2']
          })

    filename = _TMPDIR + "/draw_rectangles.png"
    cv2.imwrite(filename, canvas[:, :, ::-1])  # RGB to BGR.
    tf.logging.info("The image with rectangle is written to %s.", filename)

  def test_draw_caption(self):
    image = tf.placeholder(tf.uint8, shape=[None, None, 3])
    caption = tf.placeholder(tf.string, shape=[])

    canvas = plotlib.draw_caption(
        tf.expand_dims(image, axis=0),
        tf.expand_dims(caption, axis=0),
        org=(20, 20),
        color=(0, 0, 255),
        thickness=1,
        fontscale=1.0)

    with self.test_session() as sess:
      canvas = sess.run(
          canvas[0],
          feed_dict={
              image: np.zeros((480, 640, 3), dtype=np.uint8),
              caption: 'bye bye, world!'
          })

    filename = _TMPDIR + "/draw_caption.png"
    cv2.imwrite(filename, canvas[:, :, ::-1])  # RGB to BGR.
    tf.logging.info("The image with caption is written to %s.", filename)


if __name__ == '__main__':
  tf.test.main()
