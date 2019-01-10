from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from core import imgproc

from protos import image_resizer_pb2


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

  if 'fixed_shape_resizer' == image_resizer_oneof:
    options = options.fixed_shape_resizer

    def _fixed_shape_resize_fn(image):
      return imgproc.resize_image_to_size(
          image, new_height=options.height, new_width=options.width)

    return _fixed_shape_resize_fn

  if 'keep_aspect_ratio_resizer' == image_resizer_oneof:
    options = options.keep_aspect_ratio_resizer

    def _keep_aspect_ratio_resize_fn(image):
      return imgproc.resize_image_to_max_dimension(
          image,
          max_dimension=options.max_dimension,
          pad_to_max_dimension=options.pad_to_max_dimension)

    return _keep_aspect_ratio_resize_fn

  raise ValueError('Invalid resizer: {}.'.format(optimizer))
