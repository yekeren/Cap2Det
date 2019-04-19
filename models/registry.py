from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


_registry = {}


def register_model_class(cid, cls):
  """Registers a model class.

  Args:
    cis: Class id.
    cls: A specific class.
  """
  global _registry
  _registry[cid] = cls

  tf.logging.info('Function registered: %i', cid)


def get_registered_model_classes():
  """Returns the dict mapping class ids to classes.

  Returns:
    registry: A python dict.
  """
  return _registry
