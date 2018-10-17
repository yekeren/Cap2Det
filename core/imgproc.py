
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
        tf.nn.conv2d(channel_image, kernel, [1, 1, 1, 1], 
          padding='SAME', data_format="NHWC", name="gaussian_filter"))

  return tf.concat(outputs, axis=-1)


def calc_integral_image(image):
  """Computes the integral image.

  Args:
    image: 3-D float `Tensor` of size [b, n, m].

  Returns:
    cumsum: 3-D float `Tensor` of size [b, n + 1, m + 1]
  """
  b, n, m = utils.get_tensor_shape(image)

  pad_top = tf.fill([b, 1, m], 0.0)
  pad_left = tf.fill([b, n + 1, 1], 0.0)
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
    image: 3-D float `Tensor` of size [b, n, m].
    box: 3-D int64 `Tensor` of size [b, p, 4], representing `b` examples each 
      with `p` proposals in the format of [ymin, xmin, ymax, xmax].

  Returns:
    cumsum: 2-D float `Tensor` of size [b, p].
  """
  b, n, m = utils.get_tensor_shape(image)
  _, p, _ = utils.get_tensor_shape(box)

  cumsum = calc_integral_image(image)
  ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)

  i = tf.range(b, dtype=tf.int64)
  i = tf.tile(tf.expand_dims(i, axis=-1), [1, p])

  i_a = tf.gather_nd(cumsum, tf.stack([i, ymin, xmin], axis=-1))
  i_b = tf.gather_nd(cumsum, tf.stack([i, ymin, xmax], axis=-1))
  i_c = tf.gather_nd(cumsum, tf.stack([i, ymax, xmin], axis=-1))
  i_d = tf.gather_nd(cumsum, tf.stack([i, ymax, xmax], axis=-1))

  return i_d + i_a - i_b - i_c


def calc_box_saliency(saliency_map, box, border_ratio=0.1, alpha=1.0):
  """Computes the saliency score of each box.

  Args:
    saliency_map: 3-D float `Tensor` of size [b, n, m].
    box: 3-D int64 `Tensor` of size [b, p, 4], representing `b` examples each 
      with `p` proposals in the format of [ymin, xmin, ymax, xmax].
    border_ratio: width of the border, meatured in percentage.
    alpha: hyperparamter to balance the intensities inside and outside.

  Returns:
    box_saliency: 2-D float `Tensor` of size [b, p].
  """
  b, n, m = utils.get_tensor_shape(saliency_map)
  _, p, _ = utils.get_tensor_shape(box)

  ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
  (height_entire, width_entire) = ymax - ymin, xmax - xmin

  border_y = tf.cast(
      tf.cast(height_entire, tf.float32) * border_ratio, tf.int64)
  border_x = tf.cast(
      tf.cast(width_entire, tf.float32) * border_ratio, tf.int64)

  height_inside = height_entire - 2 * border_y
  width_inside = width_entire - 2 * border_x

  assert_op = tf.Assert(
      tf.reduce_all(tf.logical_and(height_inside > 0, width_inside > 0)), 
      ["Invalid box", height_inside, width_inside])

  with tf.control_dependencies([assert_op]):
    box_inside = tf.stack([
        ymin + border_y,
        xmin + border_x,
        ymax - border_y,
        xmax - border_x], axis=-1)
    cumsum = calc_cumsum_2d(
        saliency_map, tf.concat([box, box_inside], axis=1))
    cumsum_entire, cumsum_inside = cumsum[:, :p], cumsum[:, p:]

    (area_entire, area_inside
     ) = (height_entire * width_entire, height_inside * width_inside)

    avg_inside = tf.div(cumsum_inside, 
        _SMALL_NUMBER + tf.cast(area_inside, tf.float32))
    avg_border = tf.div(cumsum_entire - cumsum_inside, 
        _SMALL_NUMBER + tf.cast(area_entire - area_inside, tf.float32))

  return avg_inside - alpha * avg_border

  # return tf.where(valid_inside, 
  #     avg_inside - alpha * avg_border, tf.fill([b, p], -_BIG_NUMBER))


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

  num_boxes = len(boxes)
  if 0 == num_boxes:
    return 0, np.zeros((max_num_boxes, 4), dtype=np.float32)

  x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  boxes = np.stack([
      y / height, 
      x / width, 
      (y + h) / height, 
      (x + w) / width], axis=-1).astype(np.float32)

  boxes = np.concatenate(
      [boxes, np.zeros((max_num_boxes - len(boxes), 4), dtype=np.float32)],
      axis=0)
  return num_boxes, boxes


def get_edge_boxes(image, edge_detection, edge_boxes, max_num_boxes=50):
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
        func=lambda x: _py_get_edge_boxes(x, edge_detection, edge_boxes, max_num_boxes),
        inp=[image], Tout=[tf.int64, tf.float32])

    num_boxes.set_shape(tf.TensorShape([]))
    boxes.set_shape(tf.TensorShape([max_num_boxes, 4]))
    return [num_boxes, boxes]

  return tf.map_fn(_get_fn, elems=image, dtype=[tf.int64, tf.float32])
