from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf


class ModelBase(abc.ABC):
  """Model interface."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: the actual model proto.
      is_training: if True, training graph will be built.
    """
    self._model_proto = model_proto
    self._is_training = is_training

  @abc.abstractmethod
  def build_prediction(self, examples, **kwargs):
    """Builds tf graph for prediction.

    Args:
      examples: dict of input tensors keyed by name.

    Returns:
      predictions: dict of prediction results keyed by name.
    """
    pass

  @abc.abstractmethod
  def build_loss(self, predictions, **kwargs):
    """Build tf graph to compute loss.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      loss_dict: dict of loss tensors keyed by name.
    """
    pass

  @abc.abstractmethod
  def build_evaluation(self, predictions, **kwargs):
    """Build tf graph to evaluate the model.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      eval_metric_ops: dict of metric results keyed by name. The values are the
        results of calling a metric function, namely a (metric_tensor, 
        update_op) tuple. see tf.metrics for details.
    """
    pass

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      a list of model variables or None by default.
    """
    return tf.trainable_variables()

  def get_scaffold(self):
    """Returns scaffold object used to initialize variables.

    Returns:
      a tf.train.Scaffold instance or None by default.
    """
    return tf.train.Scaffold()
