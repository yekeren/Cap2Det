from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from core import imgproc
from core import utils

from protos import image_resizer_pb2
from protos import post_process_pb2

from object_detection.core.post_processing import batch_multiclass_non_max_suppression


def build_post_processor(options):
  """Builds detection post processor function.

  Args:
    options: A post_process_pb2.PostProcess instance.

  Returns:
    A callable used to process NMS.

  Raises:
    ValueError: If the options is invalid.
  """
  if not isinstance(options, post_process_pb2.PostProcess):
    raise ValueError(
        'The options has to be an instance of post_process_pb2.PostProcess.')

  def _post_process(boxes, scores, additional_fields=None):
    """Applies post process to get the final detections.

    Args:
      boxes: A [batch_size, num_anchors, q, 4] float32 tensor containing
        detections. If `q` is 1 then same boxes are used for all classes
          otherwise, if `q` is equal to number of classes, class-specific boxes
          are used.
      scores: A [batch_size, num_anchors, num_classes] float32 tensor containing
        the scores for each of the `num_anchors` detections. The scores have to be
        non-negative when use_static_shapes is set True.

    Returns:
      num_detections: A [batch_size] int32 tensor indicating the number of
        valid detections per batch item. Only the top num_detections[i] entries in
        nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
        entries are zero paddings.
      nmsed_boxes: A [batch_size, max_detections, 4] float32 tensor
        containing the non-max suppressed boxes.
      nmsed_scores: A [batch_size, max_detections] float32 tensor containing
        the scores for the boxes.
      nmsed_classes: A [batch_size, max_detections] float32 tensor
        containing the class for boxes.
    """
    boxes = tf.expand_dims(boxes, axis=2)
    (nmsed_boxes, nmsed_scores, nmsed_classes, _, nmsed_additional_fields,
     num_detections) = batch_multiclass_non_max_suppression(
         boxes,
         scores,
         score_thresh=options.score_thresh,
         iou_thresh=options.iou_thresh,
         max_size_per_class=options.max_size_per_class,
         max_total_size=options.max_total_size,
         additional_fields=additional_fields)
    return num_detections, nmsed_boxes, nmsed_scores, nmsed_classes + 1, nmsed_additional_fields

  return _post_process


def build_image_resizer(options):
  """Builds image resizing function.

  Args:
    options: An image_resizer_pb2.ImageResizer instance.

  Returns:
    A callable that takes [height, width, 3] image as input.

  Raises:
    ValueError: If the options is invalid.
  """
  if not isinstance(options, image_resizer_pb2.ImageResizer):
    raise ValueError(
        'The options has to be an instance of image_resizer_pb2.ImageResizer.')

  image_resizer_oneof = options.WhichOneof('image_resizer_oneof')

  if 'default_resizer' == image_resizer_oneof:

    def _default_resize_fn(image):
      image_shape = utils.get_tensor_shape(image)
      return tf.cast(image, tf.float32), image_shape

    return _default_resize_fn

  if 'fixed_shape_resizer' == image_resizer_oneof:
    options = options.fixed_shape_resizer

    def _fixed_shape_resize_fn(image):
      return imgproc.resize_image_to_size(
          image, new_height=options.height, new_width=options.width)

    return _fixed_shape_resize_fn

  if 'keep_aspect_ratio_resizer' == image_resizer_oneof:
    options = options.keep_aspect_ratio_resizer

    def _keep_aspect_ratio_resize_fn(image):
      return imgproc.resize_image_to_min_dimension(
          image, min_dimension=options.min_dimension)

    return _keep_aspect_ratio_resize_fn

  #if 'random_scale_resizer' == image_resizer_oneof:
  #  options = options.random_scale_resizer

  #  def _random_scale_resize_fn(image):
  #    index = tf.random_uniform(
  #        shape=[], maxval=len(options.max_dimension), dtype=tf.int32)
  #    max_dimension = tf.constant([v for v in options.max_dimension],
  #                                dtype=tf.int32)
  #    max_dimension = max_dimension[index]
  #    return imgproc.resize_image_to_max_dimension(
  #        image, max_dimension=max_dimension, pad_to_max_dimension=False)

  #  return _random_scale_resize_fn

  raise ValueError('Invalid resizer: {}.'.format(optimizer))
