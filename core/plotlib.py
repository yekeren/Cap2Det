
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from core import utils

_SMALL_NUMBER = 1e-8
_CMAP = "jet"


def _py_convert_to_heatmap(image, normalize=True):
  """Converts single-channel image to heat-map image.

  Args:
    image: a [height, width] float numpy array.
    normalize: if True, normalize the pixel values to the range of [0.0, 1.0].

  Returns:
    heatmap: a [height, width, 3] float numpy array, the values are in the
      range of [0.0, 1.0].
  """
  if normalize:
    min_v, max_v = image.min(), image.max()
    image = (image - min_v) / (_SMALL_NUMBER + max_v - min_v)

  cm = plt.get_cmap(_CMAP)
  heatmap = cm(image)
  return heatmap[:, :, :3].astype(np.float32)


def convert_to_heatmap(image, normalize=True):
  """Converts the single-channel image to heat-map image.

  Args:
    image: a [batch, height, width] float tensor.
    normalize: if True, normalize the values to the range of [0.0, 1.0].

  Returns:
    heatmap: a [batch, height, width, 3] float tensor, the values are in the
      range of [0.0, 1.0].
  """

  def _convert_fn(image):
    """Converts the single-channel image to heat-map image.

    Args:
      image: a [height, width] float tensor.
      
    Returns:
      heatmap: a [height, width, 3] float tensor.
    """
    heatmap = tf.py_func(
        func=lambda x: _py_convert_to_heatmap(x, normalize),
        inp=[image], Tout=tf.float32)

    heatmap.set_shape(tf.TensorShape([None, None, 3]))
    return heatmap

  return tf.map_fn(_convert_fn, elems=image, dtype=tf.float32)


def gaussian_kernel(ksize=3, sigma=-1.0):
  """Returns a 2D Gaussian kernel.

  See cv2.getGaussianKernel for details.

  Args:
    ksize: aperture size, it should be odd and positive.
    sigma: Gaussian standard deviation. If it is non-positive, it is computed
      from ksize as sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8.

  Returns:
    kernel: the Gaussian kernel, a 2-D tensor with shape [size, size].
  """
  kernel = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F)
  return kernel * kernel.transpose()


def gaussian_filter(inputs, ksize=3):
  """Applies Gaussian filter to the inputs.

  Args:
    inputs: input images, a [batch, height, width, channels] float tensor.
    ksize: aperture size of the Gaussian kernel.

  Returns:
    outputs: output images, a [batch, height, width, channels] float tensor.
  """
  batch, height, width, channels = utils.get_tensor_shape(inputs)

  kernel = gaussian_kernel(ksize)
  kernel = tf.reshape(tf.constant(kernel), [ksize, ksize, 1, 1])

  outputs = []
  channel_images = tf.split(inputs,
      num_or_size_splits=channels, axis=-1)

  for channel_image in channel_images:
    outputs.append( 
        tf.nn.conv2d(channel_image, kernel, [1, 1, 1, 1], 
          padding='SAME', data_format="NHWC", name="gaussian_filter"))

  return tf.concat(outputs, axis=-1)
