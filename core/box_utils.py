from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


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
  """
  with tf.name_scope('box_iou'):
    inter = area(intersect(box1, box2))
    union = area(box1) + area(box2) - inter
    iou_v = inter / union
  return iou_v
