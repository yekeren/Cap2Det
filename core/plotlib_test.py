
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from core import plotlib

_TMP = "/tmp"

tf.logging.set_verbosity(tf.logging.INFO)


class PlotLibTest(tf.test.TestCase):

  def test_py_convert_to_heatmap(self):
    kernel = plotlib.gaussian_kernel(ksize=100)
    kernel_heatmap = plotlib._py_convert_to_heatmap(kernel, normalize=True)
    kernel_heatmap = (kernel_heatmap * 255).astype(np.uint8)

    filename = _TMP + "/py_convert_to_heatmap.png"
    cv2.imwrite(filename, kernel_heatmap[:, :, ::-1])  # RGB to BGR.
    tf.logging.info("The kernel image is written to %s.", filename)

  def test_convert_to_heatmap(self):
    image = tf.placeholder(tf.float32, shape=[None, None])
    heatmap = plotlib.convert_to_heatmap(
        tf.expand_dims(image, 0), normalize=True)

    with self.test_session() as sess:
      kernel = plotlib.gaussian_kernel(ksize=100)
      kernel_heatmap = sess.run(heatmap[0], feed_dict={ image: kernel })
    kernel_heatmap = (kernel_heatmap * 255).astype(np.uint8)

    filename = _TMP + "/convert_to_heatmap.png"
    cv2.imwrite(filename, kernel_heatmap[:, :, ::-1])  # RGB to BGR.
    tf.logging.info("The kernel image is written to %s.", filename)

  def test_gaussian_kernel(self):

    # 3x3 kernel.

    kernel = plotlib.gaussian_kernel(3)
    self.assertNear(kernel.sum(), 1.0, err=1e-8)
    self.assertEqual(kernel.shape, (3, 3))
    self.assertEqual(kernel[1, 1], kernel.max())

    # 5x5 kernel.

    kernel = plotlib.gaussian_kernel(5)
    self.assertNear(kernel.sum(), 1.0, err=1e-8)
    self.assertEqual(kernel.shape, (5, 5))
    self.assertEqual(kernel[2, 2], kernel.max())

  def test_gaussian_filter(self):

    image = tf.placeholder(tf.float32, shape=[None, None, None, 1])
    outputs = plotlib.gaussian_filter(image, ksize=30)

    with self.test_session() as sess:

      # Print smoothed image.

      image_data = np.zeros((1, 10, 10, 1), dtype=np.float32)
      image_data[:, 3:7, 3:7, :] = 1.0

      image_smoothed = sess.run(outputs[0, :, :, 0], 
          feed_dict={ image: image_data })
      tf.logging.info("\n%s", image_smoothed)


if __name__ == '__main__':
  tf.test.main()
