from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
from core import utils

_BIG_NUMBER = 1e10
_SMALL_NUMBER = 1e-10


def _py_gaussian_kernel(ksize=3, sigma=-1.0):
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


def _py_get_edge_boxes(image, edge_detection, edge_boxes, max_num_boxes=50):
  """Extracts edge boxes from an image.

  Args:
    image: a [height, width, 3] float numpy array, RGB format, [0.0, 255.0].
    edge_detection: an instance of cv2.ximgproc.StructuredEdgeDetection.
    edge_boxes: an instance of cv2.ximgproc.EdgeBoxes.
    max_num_boxes: maximum number of boxes.

  Returns:
    num_boxes: a scalar int representing the number of boxes in the image.
    boxes: a [max_num_boxes, 4] float numpy array representing normalized boxes
      [ymin, xmin, ymax, xmax].
  """
  height, width, _ = image.shape
  edge_boxes.setMaxBoxes(max_num_boxes)

  edges = edge_detection.detectEdges(image / 255.0)
  orimap = edge_detection.computeOrientation(edges)

  nmsed_edges = edge_detection.edgesNms(edges, orimap)
  boxes = edge_boxes.getBoundingBoxes(nmsed_edges, orimap)

  default_box = np.array([[0, 0, 1, 1]], dtype=np.float32)

  num_boxes = len(boxes)
  if 0 == num_boxes:
    return 0, np.tile(default_box, reps=[max_num_boxes, 1])

  x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  boxes = np.stack([y / height, x / width, (y + h) / height, (x + w) / width],
                   axis=-1).astype(np.float32)

  boxes = np.concatenate(
      [boxes, np.tile(default_box, reps=[max_num_boxes - num_boxes, 1])],
      axis=0)
  return num_boxes, boxes


def gaussian_filter(inputs, ksize=3):
  """Applies Gaussian filter to the inputs.

  Args:
    inputs: input images, a [batch, height, width, channels] float tensor.
    ksize: aperture size of the Gaussian kernel.

  Returns:
    outputs: output images, a [batch, height, width, channels] float tensor.
  """
  batch, height, width, channels = utils.get_tensor_shape(inputs)

  kernel = _py_gaussian_kernel(ksize)
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


def calc_integral_image(image):
  """Computes the integral image.

  Args:
    image: 4-D float `Tensor` of size [b, n, m, c], representing `b` images with
      height `n`, width `m`, and channels `c`.

  Returns:
    cumsum: 3-D float `Tensor` of size [b, n + 1, m + 1, c], channel-wise
      cumulative sum.
  """
  b, n, m, c = utils.get_tensor_shape(image)

  pad_top = tf.fill([b, 1, m, c], 0.0)
  pad_left = tf.fill([b, n + 1, 1, c], 0.0)
  image = tf.concat([pad_top, image], axis=1)
  image = tf.concat([pad_left, image], axis=2)

  cumsum = tf.cumsum(image, axis=2)
  cumsum = tf.cumsum(cumsum, axis=1)
  return cumsum


def calc_cumsum_2d(image, box):
  """Computes the cumulative sum give pre-defiend boxes.

  i_a (ymin, xmin), ..., i_b (ymin, xmax)
  i_c (ymax, xmin), ..., i_d (ymax, xmax)

  Args:
    image: 4-D float `Tensor` of size [b, n, m, c], representing `b` images with
      height `n`, width `m`, and channels `c`.
    box: 3-D int64 `Tensor` of size [b, p, 4], representing `b` examples each 
      with `p` proposals in the format of [ymin, xmin, ymax, xmax].

  Returns:
    cumsum: 3-D float `Tensor` of size [b, p, c], channel-wise cumulative sum.
  """
  b, n, m, c = utils.get_tensor_shape(image)
  _, p, _ = utils.get_tensor_shape(box)

  cumsum = calc_integral_image(image)
  ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)

  i = tf.range(tf.cast(b, tf.int64), dtype=tf.int64)
  i = tf.tile(tf.expand_dims(i, axis=-1), [1, p])

  i_a = tf.gather_nd(cumsum, tf.stack([i, ymin, xmin], axis=-1))
  i_b = tf.gather_nd(cumsum, tf.stack([i, ymin, xmax], axis=-1))
  i_c = tf.gather_nd(cumsum, tf.stack([i, ymax, xmin], axis=-1))
  i_d = tf.gather_nd(cumsum, tf.stack([i, ymax, xmax], axis=-1))

  return i_d + i_a - i_b - i_c


def get_edge_boxes(image, edge_detection, edge_boxes, max_num_boxes):
  """Extracts edge boxes from image tensor.

  Args:
    image: a [batch, height, width, 3] float tensor, RGB format, [0.0, 255.0].
    edge_detection: an instance of cv2.ximgproc.StructuredEdgeDetection.
    edge_boxes: an instance of cv2.ximgproc.EdgeBoxes.
    max_num_boxes: maximum number of boxes.

  Returns:
    num_boxes: a [batch, num_boxes] int tensor representing number of boxes for
      each image.
    boxes: a [batch, max_num_boxes, 4] float tensor representing normalized
      boxes [ymin, xmin, ymax, xmax].
  """

  def _get_fn(image):
    """Extracts edge boxes from image tensor.

    Args:
      image: a [height, width, 3] float tensor.

    Returns:
      num_boxes: a int tensor denoting number of boxes.
      boxes: a [num_boxes, 4] tensor denoting the boxes.
    """
    num_boxes, boxes = tf.py_func(
        func=
        lambda x: _py_get_edge_boxes(x, edge_detection, edge_boxes, max_num_boxes),
        inp=[image],
        Tout=[tf.int64, tf.float32])

    num_boxes.set_shape(tf.TensorShape([]))
    boxes.set_shape(tf.TensorShape([max_num_boxes, 4]))
    return [num_boxes, boxes]

  return tf.map_fn(_get_fn, elems=image, dtype=[tf.int64, tf.float32])


def resize_image_to_size(image,
                         new_height=600,
                         new_width=1024,
                         method=tf.image.ResizeMethod.BILINEAR,
                         align_corners=False):
  """Resizes images to the given height and width.

  Args:
    image: A 3D tensor of shape [height, width, channels]
    new_height: (optional) (scalar) desired height of the image.
    new_width: (optional) (scalar) desired width of the image.
    method: (optional) interpolation method used in resizing. Defaults to 
      BILINEAR.
    align_corners: bool. If true, exactly align all 4 corners of the input
      and output. Defaults to False.

  Returns:
    resized_image: A tensor of size [new_height, new_width, channels].
    resized_image_shape: A 1D tensor of shape [3] containing the shape of the
      resized image.
  """
  with tf.name_scope("resize_image_to_size"):
    new_image = tf.image.resize_images(
        image,
        tf.stack([new_height, new_width]),
        method=method,
        align_corners=align_corners)
    image_shape = utils.get_tensor_shape(image)
    return new_image, tf.stack([new_height, new_width, image_shape[2]])


def resize_image_to_max_dimension(image,
                                  max_dimension=None,
                                  method=tf.image.ResizeMethod.BILINEAR,
                                  align_corners=False,
                                  pad_to_max_dimension=False,
                                  per_channel_pad_value=(0, 0, 0)):
  """Resizes an image so its dimensions are within the provided value.

  Args:
    image: A 3D tensor of shape [height, width, channels]
    max_dimension: (optional) (scalar) maximum allowed size
      of the larger image dimension.
    method: (optional) interpolation method used in resizing. Defaults to
      BILINEAR.
    align_corners: bool. If true, exactly align all 4 corners of the input
      and output. Defaults to False.
    pad_to_max_dimension: Whether to resize the image and pad it with zeros
      so the resulting image is of the spatial size
      [max_dimension, max_dimension].
    per_channel_pad_value: A tuple of per-channel scalar value to use for
      padding. By default pads zeros.

  Returns:
    resized_image: A 3D tensor of shape [new_height, new_width, channels],
      where the image has been resized (with bilinear interpolation) so that
      max(new_height, new_width) == max_dimension.
    resized_image_shape: A 1D tensor of shape [3] containing shape of the
      resized image.

  Raises:
    ValueError: if the image is not a 3D tensor.
  """

  def _compute_new_dynamic_size(image, max_dimension):
    """Compute new dynamic shape for resize_image_to_max_dimesion method."""

    image_shape = tf.shape(image)
    orig_height = tf.to_float(image_shape[0])
    orig_width = tf.to_float(image_shape[1])

    orig_max_dim = tf.maximum(orig_height, orig_width)
    scale_factor = tf.cast(max_dimension, dtype=tf.float32) / orig_max_dim

    new_height = tf.to_int32(tf.round(orig_height * scale_factor))
    new_width = tf.to_int32(tf.round(orig_width * scale_factor))
    new_size = tf.stack([new_height, new_width, image_shape[2]])

    return new_size

  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  with tf.name_scope("resize_image_to_range"):
    new_size = _compute_new_dynamic_size(image, max_dimension)
    new_image = tf.image.resize_images(
        image, new_size[:-1], method=method, align_corners=align_corners)

    if pad_to_max_dimension:
      channels = tf.unstack(new_image, axis=2)
      if len(channels) != len(per_channel_pad_value):
        raise ValueError(
            'Number of channels must be equal to the length of per-channel pad value.'
        )
      new_image = tf.stack([
          tf.pad(
              channels[i], [[0, max_dimension - new_size[0]],
                            [0, max_dimension - new_size[1]]],
              constant_values=per_channel_pad_value[i])
          for i in range(len(channels))
      ],
                           axis=2)
      new_image.set_shape([max_dimension, max_dimension, 3])

    return new_image, new_size


def resize_image_to_min_dimension(image,
                                  min_dimension=None,
                                  method=tf.image.ResizeMethod.BILINEAR,
                                  align_corners=False,
                                  per_channel_pad_value=(0, 0, 0)):
  """Resizes an image so its dimensions are within the provided value.

  Args:
    image: A 3D tensor of shape [height, width, channels]
    min_dimension: (optional) (scalar) minimum allowed size
      of the larger image dimension.
    method: (optional) interpolation method used in resizing. Defaults to
      BILINEAR.
    align_corners: bool. If true, exactly align all 4 corners of the input
      and output. Defaults to False.
    per_channel_pad_value: A tuple of per-channel scalar value to use for
      padding. By default pads zeros.

  Returns:
    resized_image: A 3D tensor of shape [new_height, new_width, channels],
      where the image has been resized (with bilinear interpolation) so that
      min(new_height, new_width) == min_dimension.
    resized_image_shape: A 1D tensor of shape [3] containing shape of the
      resized image.

  Raises:
    ValueError: if the image is not a 3D tensor.
  """

  def _compute_new_dynamic_size(image, min_dimension):
    """Compute new dynamic shape for resize_image_to_min_dimesion method."""

    image_shape = tf.shape(image)
    orig_height = tf.to_float(image_shape[0])
    orig_width = tf.to_float(image_shape[1])

    orig_min_dim = tf.minimum(orig_height, orig_width)
    scale_factor = tf.cast(min_dimension, dtype=tf.float32) / orig_min_dim

    new_height = tf.to_int32(tf.round(orig_height * scale_factor))
    new_width = tf.to_int32(tf.round(orig_width * scale_factor))
    new_size = tf.stack([new_height, new_width, image_shape[2]])

    return new_size

  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  with tf.name_scope("resize_image_to_range"):
    new_size = _compute_new_dynamic_size(image, min_dimension)
    new_image = tf.image.resize_images(
        image, new_size[:-1], method=method, align_corners=align_corners)

    return new_image, new_size
