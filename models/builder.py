from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import model_pb2
from protos import cap2det_model_pb2
from models.registry import get_registered_model_classes

import models.cap2det_model


def build(options, is_training=False):
  """Builds a Model based on the options.

  Args:
    options: a model_pb2.Model instance.
    is_training: True if this model is being built for training.

  Returns:
    a Model instance.

  Raises:
    ValueError: if options is invalid.
  """
  if not isinstance(options, model_pb2.Model):
    raise ValueError('The options has to be an instance of model_pb2.Model.')

  lookup_table = get_registered_model_classes()

  for extension, value in options.ListFields():
    if extension in lookup_table:
      return lookup_table[extension](value, is_training)

  raise ValueError(
      'Unknown model {}, did you forget to call register_model_class?'.format(
          extension.full_name))
