from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def scale_to_new_size(box, img_shape, pad_shape):
  """Scales to new image size.

  Args:
    box: A [batch, 4] float tensor.
    img_shape: A [2] int tensor.
    pad_shape: A [2] int tensor.
  """
  img_h, img_w = img_shape[0], img_shape[1]
  pad_h, pad_w = pad_shape[0], pad_shape[1]

  ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
  ymin = ymin * tf.to_float(img_h) / tf.to_float(pad_h)
  xmin = xmin * tf.to_float(img_w) / tf.to_float(pad_w)
  ymax = ymax * tf.to_float(img_h) / tf.to_float(pad_h)
  xmax = xmax * tf.to_float(img_w) / tf.to_float(pad_w)

  return tf.stack([ymin, xmin, ymax, xmax], axis=-1)


def flip_left_right(box):
  """Flips box left and right.

  Args:
    box: A [batch, 4] float tensor.

  Returns:
    flipped_box: A [batch, 4] float tensor.
  """
  with tf.name_scope('box_flip_left_right'):
    ymin, xmin, ymax, xmax = [box[:, i] for i in range(4)]
    box = tf.stack([ymin, 1.0 - xmax, ymax, 1.0 - xmin], axis=-1)
  return box


def area(box):
  """Compute the area of the box.

  Args:
    box: A [batch, 4] float tensor.

  Returns:
    area: The areas of the box.
  """
  with tf.name_scope('box_area'):
    ymin, xmin, ymax, xmax = [box[:, i] for i in range(4)]
    area = tf.multiply(
        tf.maximum(xmax - xmin, 0.0), tf.maximum(ymax - ymin, 0.0))
  return area


def intersect(box1, box2):
  """Compute the intersect box of the two. 

  Args:
    box1: A [batch, 4] float tensor.
    box2: A [batch, 4] float tensor.

  Returns:
    A [batch, 4] float tensor.
  """
  with tf.name_scope('box_itersect'):
    ymin1, xmin1, ymax1, xmax1 = [box1[:, i] for i in range(4)]
    ymin2, xmin2, ymax2, xmax2 = [box2[:, i] for i in range(4)]

    ymin = tf.maximum(ymin1, ymin2)
    xmin = tf.maximum(xmin1, xmin2)
    ymax = tf.minimum(ymax1, ymax2)
    xmax = tf.minimum(xmax1, xmax2)

    box = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
  return box


def iou(box1, box2):
  """Computes the Intersection-over-Union between the box1 and box2.

  Args:
    box1: A [batch, 4] float tensor.
    box2: A [batch, 4] float tensor.

  Returns:
    iou: A [batch] float tensor.
  """
  with tf.name_scope('box_iou'):
    inter = area(intersect(box1, box2))
    union = area(box1) + area(box2) - inter
    iou_v = inter / union
  return iou_v


def py_area(box):
  """Compute the area of the box.

  Args:
    box: A [batch, 4] float np array.

  Returns:
    area: The areas of the box.
  """
  ymin, xmin, ymax, xmax = [box[:, i] for i in range(4)]
  area = np.multiply(np.maximum(xmax - xmin, 0.0), np.maximum(ymax - ymin, 0.0))
  return area


def py_intersect(box1, box2):
  """Compute the intersect box of the two. 

  Args:
    box1: A [batch, 4] float np array.
    box2: A [batch, 4] float np array.

  Returns:
    A [batch, 4] float tensor.
  """
  ymin1, xmin1, ymax1, xmax1 = [box1[:, i] for i in range(4)]
  ymin2, xmin2, ymax2, xmax2 = [box2[:, i] for i in range(4)]

  ymin = np.maximum(ymin1, ymin2)
  xmin = np.maximum(xmin1, xmin2)
  ymax = np.minimum(ymax1, ymax2)
  xmax = np.minimum(xmax1, xmax2)

  box = np.stack([ymin, xmin, ymax, xmax], axis=-1)
  return box


def py_iou(box1, box2):
  """Computes the Intersection-over-Union between the box1 and box2.

  Args:
    box1: A [batch, 4] float np array.
    box2: A [batch, 4] float np array.

  Returns:
    iou: A [batch] float np array.
  """
  inter = py_area(py_intersect(box1, box2))
  union = py_area(box1) + py_area(box2) - inter
  iou_v = inter / union
  return iou_v


def py_evaluate_precision_and_recall(num_gt_boxes,
                                     gt_boxes,
                                     gt_labels,
                                     num_dt_boxes,
                                     dt_boxes,
                                     dt_labels,
                                     iou_threshold=0.5):
  """Evaluates the detection precision.

  Args:
    num_gt_boxes: Number of ground-truth boxes.
    gt_boxes: Ground-truth boxes.
    gt_labels: Ground-truth labels.
    num_dt_boxes: Number of detection boxes.
    dt_boxes: detection boxes.
    dt_labels: detection labels.

  Returns:
    recall_mask: Boolean list of length len(dt_boxes).
    precision_mask: Boolean list of length len(dt_boxes).
  """
  recall_mask = np.zeros((len(gt_boxes)), dtype=np.bool)
  precision_mask = np.zeros((len(dt_boxes)), dtype=np.bool)

  for i in range(num_dt_boxes):
    for j in range(num_gt_boxes):
      iou_v = py_iou(
          np.expand_dims(dt_boxes[i], 0), np.expand_dims(gt_boxes[j], 0))
      if not recall_mask[j] and (
          dt_labels[i] == gt_labels[j]) and iou_v[0] > iou_threshold:
        recall_mask[j] = True
        precision_mask[i] = True

  return recall_mask, precision_mask


def py_coord_norm_to_abs(box, height, width):
  """Converts normalized coordinates to absolute coordinates.

  Args:
    boxes: A [batch, 4] numpy array.
    height: image height.
    width: image width.
  """

  ymin, xmin, ymax, xmax = [box[:, i] for i in range(4)]
  box = np.stack([ymin * height, xmin * width, ymax * height, xmax * width], axis=-1)
  return box
