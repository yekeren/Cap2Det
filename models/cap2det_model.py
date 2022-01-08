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
from models.label_extractor import build_label_extractor

slim = tf.contrib.slim


class Model(ModelBase):
  """Cap2Det model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of cap2det_model_pb2.Cap2DetModel
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, cap2det_model_pb2.Cap2DetModel):
      raise ValueError('The model_proto has to be an instance of Cap2DetModel.')

    options = model_proto

    self._midn_postprocess_fn = function_builder.build_post_processor(
        options.midn_post_processor)
    self._oicr_postprocess_fn = function_builder.build_post_processor(
        options.oicr_post_processor)

    self._label_extractor = build_label_extractor(options.label_extractor)

  def _build_midn_network(self, num_proposals, proposal_features, num_classes):
    """Builds the Multiple Instance Detection Network.

    MIDN: An attention network.

    Args:
      num_proposals: A [batch] int tensor.
      proposal_features: A [batch, max_num_proposals, features_dims] tensor.
      num_classes: Number of classes.

    Returns:
      logits: A [batch, num_classes] float tensor.
      proba_r_given_c: A [batch, max_num_proposals, num_classes] float tensor.
    """
    with tf.name_scope('multi_instance_detection'):

      batch, max_num_proposals, _ = utils.get_tensor_shape(proposal_features)
      mask = tf.sequence_mask(num_proposals,
                              maxlen=max_num_proposals,
                              dtype=tf.float32)
      mask = tf.expand_dims(mask, axis=-1)

      # Calculates the values of following tensors:
      #   logits_r_given_c shape = [batch, max_num_proposals, num_classes].
      #   logits_c_given_r shape = [batch, max_num_proposals, num_classes].

      with tf.variable_scope('midn'):
        logits_r_given_c = slim.fully_connected(proposal_features,
                                                num_outputs=num_classes,
                                                activation_fn=None,
                                                scope='proba_r_given_c')
        logits_c_given_r = slim.fully_connected(proposal_features,
                                                num_outputs=num_classes,
                                                activation_fn=None,
                                                scope='proba_c_given_r')

      # Calculates the detection scores.

      proba_r_given_c = utils.masked_softmax(data=tf.multiply(
          mask, logits_r_given_c),
                                             mask=mask,
                                             dim=1)
      proba_r_given_c = tf.multiply(mask, proba_r_given_c)

      # Aggregates the logits.

      class_logits = tf.multiply(logits_c_given_r, proba_r_given_c)
      class_logits = utils.masked_sum(data=class_logits, mask=mask, dim=1)

      proposal_scores = tf.multiply(tf.nn.sigmoid(class_logits),
                                    proba_r_given_c)

      tf.summary.histogram('midn/logits_r_given_c', logits_r_given_c)
      tf.summary.histogram('midn/logits_c_given_r', logits_c_given_r)
      tf.summary.histogram('midn/proposal_scores', proposal_scores)
      tf.summary.histogram('midn/class_logits', class_logits)

    return tf.squeeze(class_logits, axis=1), proposal_scores, proba_r_given_c

  def _postprocess(self, inputs, predictions):
    """Post processes the predictions.

    Args:
      inputs: A dict of input tensors keyed by name.
      predictions: A dict of predicted tensors.

    Returns:
      results: A dict of nmsed tensors.
    """
    results = {}
    oicr_iterations = self._model_proto.oicr_iterations

    # Post process to get the final detections.

    proposals = predictions[DetectionResultFields.proposal_boxes]

    for i in range(1 + oicr_iterations):
      post_process_fn = self._midn_postprocess_fn
      proposal_scores = tf.stop_gradient(
          predictions[Cap2DetPredictions.oicr_proposal_scores +
                      '_at_{}'.format(i)])
      if i > 0:
        post_process_fn = self._oicr_postprocess_fn
        proposal_scores = tf.nn.softmax(proposal_scores, axis=-1)[:, :, 1:]

      # NMS process.

      (num_detections, detection_boxes, detection_scores, detection_classes,
       _) = post_process_fn(proposals, proposal_scores)

      results[DetectionResultFields.num_detections +
              '_at_{}'.format(i)] = num_detections
      results[DetectionResultFields.detection_boxes +
              '_at_{}'.format(i)] = detection_boxes
      results[DetectionResultFields.detection_scores +
              '_at_{}'.format(i)] = detection_scores
      results[DetectionResultFields.detection_classes +
              '_at_{}'.format(i)] = detection_classes
    return results

  def _build_prediction(self, examples):
    """Builds tf graph for prediction.

    Args:
      examples: dict of input tensors keyed by name.
      prediction_task: the specific prediction task.

    Returns:
      predictions: dict of prediction results keyed by name.
    """
    predictions = {}
    options = self._model_proto
    is_training = self._is_training

    (inputs, num_proposals,
     proposals) = (examples[InputDataFields.image],
                   examples[InputDataFields.num_proposals],
                   examples[InputDataFields.proposals])

    # Fast-RCNN.

    proposal_features = model_utils.extract_frcnn_feature(
        inputs, num_proposals, proposals, options.frcnn_options, is_training)

    # Build MIDN network.
    #   proba_r_given_c shape = [batch, max_num_proposals, num_classes].

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      (midn_class_logits, midn_proposal_scores,
       midn_proba_r_given_c) = self._build_midn_network(
           num_proposals,
           proposal_features,
           num_classes=self._label_extractor.num_classes)

    # Build the OICR network.
    #   proposal_scores shape = [batch, max_num_proposals, 1 + num_classes].
    #   See `Multiple Instance Detection Network with OICR`.

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      for i in range(options.oicr_iterations):
        predictions[
            Cap2DetPredictions.oicr_proposal_scores +
            '_at_{}'.format(i + 1)] = proposal_scores = slim.fully_connected(
                proposal_features,
                num_outputs=1 + self._label_extractor.num_classes,
                activation_fn=None,
                scope='oicr/iter{}'.format(i + 1))

    # Set the predictions.

    predictions.update({
        DetectionResultFields.class_labels:
            tf.constant(self._label_extractor.classes),
        DetectionResultFields.num_proposals:
            num_proposals,
        DetectionResultFields.proposal_boxes:
            proposals,
        Cap2DetPredictions.midn_class_logits:
            midn_class_logits,
        Cap2DetPredictions.midn_proba_r_given_c:
            midn_proba_r_given_c,
        Cap2DetPredictions.oicr_proposal_scores + '_at_0':
            midn_proposal_scores
    })

    return predictions

  def build_prediction(self, examples, **kwargs):
    """Builds tf graph for prediction.

    Args:
      examples: dict of input tensors keyed by name.
      prediction_task: the specific prediction task.

    Returns:
      predictions: dict of prediction results keyed by name.
    """
    options = self._model_proto
    is_training = self._is_training

    if is_training or len(options.eval_min_dimension) == 0:
      predictions = self._build_prediction(examples)
      predictions.update(self._postprocess(examples, predictions))
      return predictions

    inputs = examples[InputDataFields.image]
    assert inputs.get_shape()[0].value == 1

    proposal_scores_list = [[] for _ in range(1 + options.oicr_iterations)]

    # Get predictions from different resolutions.

    reuse = False
    for min_dimension in options.eval_min_dimension:
      inputs_resized = tf.expand_dims(imgproc.resize_image_to_min_dimension(
          inputs[0], min_dimension)[0],
                                      axis=0)
      examples[InputDataFields.image] = inputs_resized

      with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        predictions = self._build_prediction(examples)

      for i in range(1 + options.oicr_iterations):
        proposals_scores = predictions[Cap2DetPredictions.oicr_proposal_scores +
                                       '_at_{}'.format(i)]
        proposal_scores_list[i].append(proposals_scores)

      reuse = True

    # Aggregate predictions from different resolutions.

    predictions_aggregated = predictions
    for i in range(1 + options.oicr_iterations):
      proposal_scores = tf.stack(proposal_scores_list[i], axis=-1)
      proposal_scores = tf.reduce_mean(proposal_scores, axis=-1)
      predictions_aggregated[Cap2DetPredictions.oicr_proposal_scores +
                             '_at_{}'.format(i)] = proposal_scores

    predictions_aggregated.update(
        self._postprocess(inputs, predictions_aggregated))

    return predictions_aggregated

  def build_loss(self, predictions, examples, **kwargs):
    """Build tf graph to compute loss.

    Args:
      predictions: dict of prediction results keyed by name.
      examples: dict of inputs keyed by name.

    Returns:
      loss_dict: dict of loss tensors keyed by name.
    """
    options = self._model_proto
    example_weight = examples[InputDataFields.example_weight]
    tf.summary.histogram('example_weight', example_weight)

    loss_dict = {}

    with tf.name_scope('losses'):

      # Loss of the MIDN module.

      labels = self._label_extractor.extract_labels(examples)
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels,
          logits=predictions[Cap2DetPredictions.midn_class_logits])
      losses = losses * tf.expand_dims(example_weight, 1)
      loss_dict['midn_cross_entropy_loss'] = tf.multiply(
          tf.reduce_mean(losses), options.midn_loss_weight)

      # Losses of the OICR module.

      (num_proposals,
       proposals) = (predictions[DetectionResultFields.num_proposals],
                     predictions[DetectionResultFields.proposal_boxes])
      batch, max_num_proposals, _ = utils.get_tensor_shape(proposals)

      proposal_scores_0 = predictions[Cap2DetPredictions.oicr_proposal_scores +
                                      '_at_0']
      if options.oicr_use_proba_r_given_c:
        proposal_scores_0 = predictions[Cap2DetPredictions.midn_proba_r_given_c]
      proposal_scores_0 = tf.concat(
          [tf.fill([batch, max_num_proposals, 1], 0.0), proposal_scores_0],
          axis=-1)

      for i in range(options.oicr_iterations):
        proposal_scores_1 = predictions[Cap2DetPredictions.oicr_proposal_scores
                                        + '_at_{}'.format(i + 1)]
        oicr_cross_entropy_losses_at_i = model_utils.calc_oicr_loss(
            labels,
            num_proposals,
            proposals,
            tf.stop_gradient(proposal_scores_0),
            proposal_scores_1,
            scope='oicr_{}'.format(i + 1),
            iou_threshold=options.oicr_iou_threshold)
        oicr_cross_entropy_losses_at_i = oicr_cross_entropy_losses_at_i * example_weight
        oicr_cross_entropy_loss_at_i = tf.reduce_mean(
            oicr_cross_entropy_losses_at_i)
        loss_dict['oicr_cross_entropy_loss_at_{}'.format(i + 1)] = tf.multiply(
            oicr_cross_entropy_loss_at_i, options.oicr_loss_weight)

        proposal_scores_0 = tf.nn.softmax(proposal_scores_1, axis=-1)

    return loss_dict

  def build_evaluation(self, predictions, examples, **kwargs):
    """Build tf graph to evaluate the model.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      eval_metric_ops: dict of metric results keyed by name. The values are the
        results of calling a metric function, namely a (metric_tensor, 
        update_op) tuple. see tf.metrics for details.
    """
    return {}


register_model_class(cap2det_model_pb2.Cap2DetModel.ext, Model)
