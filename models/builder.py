
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import model_pb2
from models import gap_model
from protos import gap_model_pb2


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

  if options.HasExtension(gap_model_pb2.GAPModel.ext):
    return gap_model.Model(
        options.Extensions[gap_model_pb2.GAPModel.ext], is_training)

  raise ValueError('Unknown model: {}'.format(model))
