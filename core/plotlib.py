from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import functools
import base64

from core import utils

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

_SMALL_NUMBER = 1e-8
_CMAP = "jet"
_FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL

# NOTE: THE DEFAULT CHANNEL ORDER IS RGB.


def _py_convert_to_base64(image, ext='.jpg', quality=90):
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
  _, encoded = cv2.imencode(ext, image, encode_param)
  return base64.encodestring(encoded.tostring()).decode('utf8').replace(
      '\n', '')


def _py_convert_to_heatmap(image, normalize=True, normalize_to=None,
                           cmap=_CMAP):
  """Converts single-channel image to heat-map image.

  Args:
    image: a [height, width] float numpy array.
    normalize: if True, normalize the pixel values to the range of [0.0, 1.0].

  Returns:
    heatmap: a [height, width, 3] float numpy array, the values are in the
      range of [0.0, 1.0].
  """
  if normalize:
    if normalize_to is None:
      min_v, max_v = image.min(), image.max()
    else:
      min_v, max_v = normalize_to
      image = np.maximum(image, min_v)
      image = np.minimum(image, max_v)
    image = (image - min_v) / (_SMALL_NUMBER + max_v - min_v)

  cm = plt.get_cmap(cmap)
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


def _py_draw_rectangles(image,
                        boxes,
                        scores,
                        labels,
                        color=GREEN,
                        thickness=1,
                        fontscale=1.0):
  """Draws boxes on the image.

  Args:
    image: a [height, width, 3] uint8 numpy array.
    boxes: a [batch, 4] float numpy array representing normalized boxes.
    scores: a [batch] float numpy array representing box scores.
    labels: a [batch] string numpy array representing labels.
    color: the color to be drawn.
    thickness: thinkness of the line.
    fontscale: scale of the font.

  Returns:
    canvas: a [height, width, 3] uint8 numpy array.
  """
  height, width, _ = image.shape

  canvas = image.copy()
  for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
    label = label.decode('UTF8') if isinstance(label, bytes) else label
    if label and score > -1000:
      text = '%s: %.3lf' % (label, score)
    elif score > -1000:
      text = '%.3lf' % (score)
    else:
      text = label
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONTFACE, fontscale,
                                                 thickness)

    ymin, xmin, ymax, xmax = box
    ymin, xmin, ymax, xmax = (int(height * ymin + 0.5), int(width * xmin + 0.5),
                              int(height * ymax + 0.5), int(width * xmax + 0.5))

    cv2.rectangle(
        canvas,
        pt1=(xmin, ymin),
        pt2=(xmax, ymax),
        color=color,
        thickness=thickness)
    if text:
      cv2.rectangle(
          canvas,
          pt1=(xmin + thickness, ymin + thickness),
          pt2=(xmin + thickness + text_w, ymin + thickness + text_h),
          color=color,
          thickness=-1)
      text_color = BLACK if color != BLACK else WHITE
      cv2.putText(
          canvas,
          text,
          org=(xmin, ymin + text_h),
          fontFace=_FONTFACE,
          fontScale=fontscale,
          color=text_color,
          thickness=thickness)
  return canvas


def _py_draw_rectangles_v2(image,
                           total,
                           boxes,
                           scores,
                           labels,
                           color=GREEN,
                           thickness=1,
                           fontscale=1.0,
                           show_score=True):
  """Draws boxes on the image.

  Args:
    image: a [height, width, 3] uint8 numpy array.
    boxes: a [batch, 4] float numpy array representing normalized boxes.
    scores: a [batch] float numpy array representing box scores.
    labels: a [batch] string numpy array representing labels.
    color: the color to be drawn.
    thickness: thinkness of the line.
    fontscale: scale of the font.

  Returns:
    canvas: a [height, width, 3] uint8 numpy array.
  """
  height, width, _ = image.shape

  canvas = image.copy()
  for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
    if i >= total: break

    label = label.decode('UTF8') if isinstance(label, bytes) else label
    text = '%s: %.0lf%%' % (label, score * 100) if show_score else label

    (text_w, text_h), baseline = cv2.getTextSize(text, _FONTFACE, fontscale,
                                                 thickness)

    ymin, xmin, ymax, xmax = box
    ymin, xmin, ymax, xmax = (int(height * ymin + 0.5), int(width * xmin + 0.5),
                              int(height * ymax + 0.5), int(width * xmax + 0.5))

    cv2.rectangle(
        canvas,
        pt1=(xmin, ymin),
        pt2=(xmax, ymax),
        color=color,
        thickness=thickness * 2)
    cv2.rectangle(
        canvas,
        pt1=(xmin, ymin),
        pt2=(xmin + 2 * thickness + text_w, ymin + 2 * thickness + text_h),
        color=color,
        thickness=-1)
    text_color = BLACK if color != BLACK else WHITE
    cv2.putText(
        canvas,
        text,
        org=(xmin, ymin + text_h),
        fontFace=_FONTFACE,
        fontScale=fontscale,
        color=text_color,
        thickness=thickness)
  return canvas


def _py_draw_caption(image,
                     caption,
                     org,
                     color=GREEN,
                     thickness=1,
                     fontscale=1.0):
  """Draws boxes on the image.

  Args:
    image: a [height, width, 3] uint8 numpy array.
    caption: a python string.
    org: the coordinate of the text.
    color: the color to be drawn.
    thickness: thinkness of the line.
    fontscale: scale of the font to be drawn.

  Returns:
    canvas: a [height, width, 3] uint8 numpy array.
  """
  height, width, _ = image.shape

  canvas = image.copy()
  caption = caption.decode('UTF8') if isinstance(caption, bytes) else caption
  (text_w, text_h), baseline = cv2.getTextSize(caption, _FONTFACE, fontscale,
                                               thickness)

  cv2.rectangle(
      canvas,
      pt1=(org[0], org[1]),
      pt2=(org[0] + text_w + thickness, org[1] + text_h + thickness),
      color=color,
      thickness=-1)
  cv2.putText(
      canvas,
      caption,
      org=(org[0], org[1] + text_h),
      fontFace=_FONTFACE,
      fontScale=fontscale,
      color=BLACK,
      thickness=thickness)
  return canvas


def convert_to_heatmap(image, normalize=True, normalize_to=None):
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
        func=lambda x: _py_convert_to_heatmap(x, normalize, normalize_to),
        inp=[image],
        Tout=tf.float32)

    heatmap.set_shape(tf.TensorShape([None, None, 3]))
    return heatmap

  return tf.map_fn(_convert_fn, elems=image, dtype=tf.float32)


def draw_caption(image, caption, org, color=GREEN, thickness=1, fontscale=1.0):
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

  def _draw_fn(inputs):
    """Draws the box on the image.

    Args:
      image: a [height, width, 3] uint8 tensor.
      caption: a scalar string tensor.

    Returns:
      canvas: a [height, width, 3] uint8 tensor with box drawn.
    """
    image, caption = inputs
    canvas = tf.py_func(
        func=
        lambda x, y: _py_draw_caption(x, y, org, color, thickness, fontscale),
        inp=[image, caption],
        Tout=tf.uint8)

    canvas.set_shape(tf.TensorShape([None, None, 3]))
    return canvas

  return tf.map_fn(_draw_fn, elems=[image, caption], dtype=tf.uint8)


def draw_rectangles(image,
                    boxes,
                    scores=None,
                    labels=None,
                    color=GREEN,
                    thickness=1,
                    fontscale=1.0):
  """Draws rectangle to the image.

  Args:
    image: a [batch, height, width, 3] uint8 tensor.
    boxes: a [batch, num_boxes, 4] float tensor representing normalized boxes,
      i.e.: [ymin, xmin, ymax, xmax], values are ranging from 0.0 to 1.0.
    scores: a [batch, num_boxes] float tensor representing the scores to be
      drawn on the image.
    labels: a [batch, num_boxes] string or float tensor representing the labels
      to be drawn on the image.
    color: color to be used.
    thickness: the line thickness.
    fontscale: size of the font.

  Returns:
    canvas: a [batch, height, width, 3] uint8 tensor with information drawn.
  """

  def _draw_fn(inputs):
    """Draws the box on the image.

    Args:
      image: a [height, width, 3] float tensor.
      box: a [num_boxes, 4] float tensor representing [ymin, xmin, ymax, xmax].
      score: a [num_boxes] float tensor representing box scores.
      label: a [num_boxes] string tensor denoting the text to be drawn.

    Returns:
      canvas: a [height, width, 3] float tensor with box drawn.
    """
    image, boxes, scores, labels = inputs
    canvas = tf.py_func(
        func=lambda x, y, z, w: _py_draw_rectangles(x, y, z, w, color=color, thickness=thickness, fontscale=fontscale),
        inp=[image, boxes, scores, labels], Tout=tf.uint8)
    canvas.set_shape(tf.TensorShape([None, None, 3]))
    return canvas

  batch, num_boxes, _ = utils.get_tensor_shape(boxes)
  if scores is None:
    scores = tf.constant(-9999.0, shape=[batch, num_boxes], dtype=tf.float32)
  if labels is None:
    labels = tf.constant("", shape=[batch, num_boxes], dtype=tf.string)

  return tf.map_fn(
      _draw_fn, elems=[image, boxes, scores, labels], dtype=tf.uint8)


def draw_rectangles_v2(image,
                       total,
                       boxes,
                       scores,
                       labels,
                       color=GREEN,
                       thickness=1,
                       fontscale=1.0):
  """Draws rectangle to the image.

  Args:
    image: a [batch, height, width, 3] uint8 tensor.
    total: a [batch] int tensor denoting number of boxes for each image.
    boxes: a [batch, num_boxes, 4] float tensor representing normalized boxes,
      i.e.: [ymin, xmin, ymax, xmax], values are ranging from 0.0 to 1.0.
    scores: a [batch, num_boxes] float tensor representing the scores to be
      drawn on the image.
    labels: a [batch, num_boxes] string or float tensor representing the labels
      to be drawn on the image.
    color: color to be used.
    thickness: the line thickness.
    fontscale: size of the font.

  Returns:
    canvas: a [batch, height, width, 3] uint8 tensor with information drawn.
  """

  def _draw_fn(inputs):
    """Draws the box on the image.

    Args:
      image: a [height, width, 3] float tensor.
      box: a [num_boxes, 4] float tensor representing [ymin, xmin, ymax, xmax].
      score: a [num_boxes] float tensor representing box scores.
      label: a [num_boxes] string tensor denoting the text to be drawn.

    Returns:
      canvas: a [height, width, 3] float tensor with box drawn.
    """
    image, total, boxes, scores, labels = inputs
    canvas = tf.py_func(
        func=lambda a, b, c, d, e: _py_draw_rectangles_v2(a, b, c, d, e, color=color, thickness=thickness, fontscale=fontscale),
        inp=[image, total, boxes, scores, labels], Tout=tf.uint8)
    canvas.set_shape(tf.TensorShape([None, None, 3]))
    return canvas

  return tf.map_fn(
      _draw_fn, elems=[image, total, boxes, scores, labels], dtype=tf.uint8)


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
  channel_images = tf.split(inputs, num_or_size_splits=channels, axis=-1)

  for channel_image in channel_images:
    outputs.append(
        tf.nn.conv2d(
            channel_image,
            kernel, [1, 1, 1, 1],
            padding='SAME',
            data_format="NHWC",
            name="gaussian_filter"))

  return tf.concat(outputs, axis=-1)
