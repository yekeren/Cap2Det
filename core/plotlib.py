
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import functools

from core import utils

_SMALL_NUMBER = 1e-8
_CMAP = "jet"

# NOTE: THE DEFAULT CHANNEL ORDER IS RGB.


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


def _py_show_heatmap(image, saliency):
  """Shows heatmap on the original image.

  Args:
    image: a [height, width, 3] uint8 numpy array.
    saliency: a [height, width] float numpy array.

  Returns:
    image_with_heatmap: a [height, width, 3] uint8 numpy array with heatmap
      visualization.
  """
  heatmap = _py_convert_to_heatmap(saliency, normalize=False)

  min_v, max_v = saliency.min(), saliency.max()
  saliency = (saliency - min_v) / (_SMALL_NUMBER + max_v - min_v)
  saliency = np.expand_dims(saliency, -1)

  image_with_heatmap = np.add(
      np.multiply(1.0 - saliency, image.astype(np.float32)),
      np.multiply(saliency, heatmap * 255.0)).astype(np.uint8)
  return image_with_heatmap


def _py_draw_rectangles(image, boxes, 
    labels=None, color=(255, 0, 0), thickness=1, fontscale=1.0):
  """Draws boxes on the image.

  Args:
    image: a [height, width, 3] uint8 numpy array.
    boxes: a [batch, 4] float numpy array representing normalized boxes.
    labels: a [batch] float or string numpy array represneting labels.
    color: the color to be drawn.
    thickness: thinkness of the line.

  Returns:
    canvas: a [height, width, 3] uint8 numpy array.
  """
  height, width, _ = image.shape
  canvas = image.copy()

  for i, box in enumerate(boxes):
    ymin, xmin, ymax, xmax = box
    cv2.rectangle(canvas, 
        pt1=(int(width * xmin + 0.5), int(height * ymin + 0.5)),
        pt2=(int(width * xmax + 0.5), int(height * ymax + 0.5)),
        color=color, thickness=thickness)
    if labels is not None:
      label = labels[i]
      cv2.putText(canvas, 
          '%.4lf' % (label) if type(label) == np.float32 else label,
          org=(int(width * xmin + 0.5) + 10, int(height * ymin + 0.5) + 35),
          fontFace=cv2.FONT_HERSHEY_SIMPLEX,
          fontScale=fontscale, 
          color=color,
          thickness=thickness)
  return canvas


def _py_draw_caption(image, caption, org, fontscale, 
    color=(255, 0, 0), thickness=1):
  """Draws boxes on the image.

  Args:
    image: a [height, width, 3] uint8 numpy array.
    caption: a python string.
    org: the coordinate of the text.
    fontscale: scale of the font to be drawn.
    color: the color to be drawn.
    thickness: thinkness of the line.

  Returns:
    canvas: a [height, width, 3] uint8 numpy array.
  """
  height, width, _ = image.shape
  canvas = image.copy()

  if type(caption) == bytes:
    caption = caption.decode('utf-8')
  cv2.putText(canvas, caption, 
      org=org, 
      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=fontscale, 
      color=color,
      thickness=thickness)
  return canvas


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


def draw_caption(image, caption, org, fontscale, color, thickness):
  """Draws caption on the image.

  Args:
    image: a [batch, height, width, 3] uint8 tensor.
    caption: a [batch] string tensor.
    org: the coordinate of the text.
    fontscale: scale of the font to be drawn.
    color: the color to be drawn.
    thickness: thinkness of the line.

  Returns:
    canvas: a [batch, height, width, 3] uint8 tensor with caption drawn.
  """
  def _draw_fn(image_and_caption):
    """Draws the box on the image.

    Args:
      image: a [height, width, 3] uint8 tensor.
      caption: a scalar string tensor.

    Returns:
      canvas: a [height, width, 3] uint8 tensor with box drawn.
    """
    image, caption = image_and_caption
    canvas = tf.py_func(
        func=lambda x, y: _py_draw_caption(x, y, org, fontscale, color, thickness),
        inp=[image, caption], Tout=tf.uint8)

    canvas.set_shape(tf.TensorShape([None, None, 3]))
    return canvas

  return tf.map_fn(_draw_fn, elems=[image, caption], dtype=tf.uint8)


def draw_rectangles(image, boxes, 
    labels=None, color=(255, 0, 0), thickness=1, fontscale=1.0):
  """Draws rectangle to the image.

  Args:
    image: a [batch, height, width, 3] uint8 tensor.
    boxes: a [batch, num_boxes, 4] float tensor representing normalized boxes,
      i.e.: [ymin, xmin, ymax, xmax], values are ranging from 0.0 to 1.0.
    labels: a [batch, num_boxes] string or float tensor representing the labels
      to be drawn on the image.
    color: color to use.
    thickness: line thickness.
  Returns:
    canvas: a [batch, height, width, 3] uint8 tensor with box drawn.
  """
  def _draw_fn(image_and_boxes):
    """Draws the box on the image.

    Args:
      image: a [height, width, 3] float tensor.
      box: a [num_boxes, 4] float tensor representing [ymin, xmin, ymax, xmax].

    Returns:
      canvas: a [height, width, 3] float tensor with box drawn.
    """
    image, boxes = image_and_boxes
    canvas = tf.py_func(
        func=lambda x, y: _py_draw_rectangles(x, y, color=color, thickness=thickness, fontscale=fontscale),
        inp=[image, boxes], Tout=tf.uint8)

    canvas.set_shape(tf.TensorShape([None, None, 3]))
    return canvas

  def _draw_fn_with_labels(image_and_boxes):
    """Draws the box on the image.

    Args:
      image: a [height, width, 3] float tensor.
      box: a [num_boxes, 4] float tensor representing [ymin, xmin, ymax, xmax].
      labels: a [num_boxes] float tensor.

    Returns:
      canvas: a [height, width, 3] float tensor with box drawn.
    """
    image, boxes, labels = image_and_boxes
    canvas = tf.py_func(
        func=lambda x, y, z: _py_draw_rectangles(x, y, z, color=color, thickness=thickness, fontscale=fontscale),
        inp=[image, boxes, labels], Tout=tf.uint8)

    canvas.set_shape(tf.TensorShape([None, None, 3]))
    return canvas

  if labels is None:
    return tf.map_fn(_draw_fn, elems=[image, boxes], dtype=tf.uint8)

  return tf.map_fn(_draw_fn_with_labels, elems=[image, boxes, labels], dtype=tf.uint8)

@utils.deprecated
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


@utils.deprecated
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
