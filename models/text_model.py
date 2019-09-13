from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from models.model_base import ModelBase
from protos import cap2det_model_pb2

from core import imgproc
from core import utils
from core import plotlib
from core.standard_fields import InputDataFields
from core.standard_fields import Cap2DetPredictions
from core.standard_fields import DetectionResultFields
from core.training_utils import build_hyperparams
from models import utils as model_utils
from core import box_utils
from core import builder as function_builder

from models.registry import register_model_class
from models import label_extractor

slim = tf.contrib.slim

FIELD_LOGITS = 'logits'
FIELD_TEXT_LOSS = 'text_cross_entropy_loss'


class Model(ModelBase):
  """Cap2Det model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of cap2det_model_pb2.TextModel
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, cap2det_model_pb2.TextModel):
      raise ValueError('The model_proto has to be an instance of TextModel.')

    options = model_proto

    self._label_extractor = label_extractor.GroundtruthExtractor(
        options.label_extractor)
    self._text_classifier = label_extractor.TextClassifierMatchExtractor(
        options.text_classifier)

  def build_prediction(self, examples, **kwargs):
    """Builds tf graph for prediction.

    Args:
      examples: dict of input tensors keyed by name.
      prediction_task: the specific prediction task.

    Returns:
      predictions: dict of prediction results keyed by name.
    """
    logits = self._text_classifier.predict(
        examples, is_training=self._is_training)
    return {FIELD_LOGITS: logits}

  def build_loss(self, predictions, examples, **kwargs):
    """Build tf graph to compute loss.

    Args:
      predictions: dict of prediction results keyed by name.
      examples: dict of inputs keyed by name.

    Returns:
      loss_dict: dict of loss tensors keyed by name.
    """
    logits = predictions[FIELD_LOGITS]
    labels = self._label_extractor.extract_labels(examples)

    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return {FIELD_TEXT_LOSS: tf.reduce_mean(losses)}

  def build_evaluation(self, predictions, examples, **kwargs):
    """Build tf graph to evaluate the model.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      eval_metric_ops: dict of metric results keyed by name. The values are the
        results of calling a metric function, namely a (metric_tensor, 
        update_op) tuple. see tf.metrics for details.
    """
    logits = predictions[FIELD_LOGITS]
    labels = self._label_extractor.extract_labels(examples)

    assert labels.get_shape()[0] == 1

    sparse_labels = tf.boolean_mask(
        tf.range(self._label_extractor.num_classes, dtype=tf.int64),
        labels[0, :] > 0)
    sparse_labels = tf.expand_dims(sparse_labels, axis=0)

    eval_metric_ops = {}
    for threshold in [0.3, 0.5, 0.7]:
      predictions = tf.nn.sigmoid(logits) > threshold

      precision, update_op = tf.metrics.precision(labels, predictions)
      eval_metric_ops['metrics/precision_at_{}'.format(threshold)] = (precision,
                                                                      update_op)

      recall, update_op = tf.metrics.recall(labels, predictions)
      eval_metric_ops['metrics/recall_at_{}'.format(threshold)] = (recall,
                                                                   update_op)

    for k in [1, 5]:
      precision, update_op = tf.metrics.precision_at_k(
          sparse_labels, logits, k=k)
      eval_metric_ops['metrics/precision_at_{}'.format(k)] = (precision,
                                                              update_op)

      recall, update_op = tf.metrics.recall_at_k(sparse_labels, logits, k=k)
      eval_metric_ops['metrics/recall_at_{}'.format(k)] = (recall, update_op)

    return eval_metric_ops


register_model_class(cap2det_model_pb2.TextModel.ext, Model)
