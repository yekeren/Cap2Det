from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from models.model_base import ModelBase
from protos import nod_model_pb2

from nets import nets_factory
from nets import vgg
from core import imgproc
from core import utils
from core import plotlib
from core.standard_fields import InputDataFields
from core.standard_fields import NODPredictions
from core.standard_fields import DetectionResultFields
from core.training_utils import build_hyperparams
from core import init_grid_anchors
from models import utils as model_utils
from core import box_utils
from core import builder as function_builder

from object_detection.builders import hyperparams_builder
from object_detection.builders.model_builder import _build_faster_rcnn_feature_extractor as build_faster_rcnn_feature_extractor

slim = tf.contrib.slim
_EPSILON = 1e-8


class Model(ModelBase):
  """NOD model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of nod_model_pb2.NODModel
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, nod_model_pb2.NODModel):
      raise ValueError('The model_proto has to be an instance of NODModel.')

    options = model_proto

    self._vocabulary_list = model_utils.read_vocabulary(options.vocabulary_file)

    self._num_classes = len(self._vocabulary_list)

    self._feature_extractor = build_faster_rcnn_feature_extractor(
        options.feature_extractor, is_training,
        options.inplace_batchnorm_update)

    self._midn_post_process_fn = function_builder.build_post_processor(
        options.midn_post_process)

    self._oicr_post_process_fn = function_builder.build_post_processor(
        options.oicr_post_process)

  def _extract_class_label(self, class_texts, vocabulary_list):
    """Extracts class labels.

    Args:
      class_texts: a [batch, 1, max_caption_len] string tensor.
      vocabulary_list: a list of words of length `num_classes`.

    Returns:
      labels: a [batch, num_classes] float tensor.
    """
    with tf.name_scope('extract_class_label'):
      batch, _, _ = utils.get_tensor_shape(class_texts)

      categorical_col = tf.feature_column.categorical_column_with_vocabulary_list(
          key='name_to_id', vocabulary_list=vocabulary_list, num_oov_buckets=1)
      indicator_col = tf.feature_column.indicator_column(categorical_col)
      indicator = tf.feature_column.input_layer({
          'name_to_id': class_texts
      },
                                                feature_columns=[indicator_col])
      labels = tf.cast(indicator[:, :-1] > 0, tf.float32)
      labels.set_shape([batch, len(vocabulary_list)])

    return labels

  def _extract_frcnn_feature(self, inputs, num_proposals, proposals):
    """Extracts Fast-RCNN feature from image.

    Args:
      inputs: A [batch, height, width, channels] float tensor.
      num_proposals: A [batch] int tensor.
      proposals: A [batch, max_num_proposals, 4] float tensor.

    Returns:
      proposal_features: A [batch, max_num_proposals, feature_dims] float 
        tensor.
    """
    options = self._model_proto
    is_training = self._is_training

    # Extract `features_to_crop` from the original image.
    #   shape = [batch, feature_height, feature_width, feature_depth].

    preprocessed_inputs = self._feature_extractor.preprocess(inputs)

    (features_to_crop, _) = self._feature_extractor.extract_proposal_features(
        preprocessed_inputs, scope='first_stage_feature_extraction')

    if options.dropout_on_feature_map:
      features_to_crop = slim.dropout(
          features_to_crop,
          keep_prob=options.dropout_keep_prob,
          is_training=is_training)

    # Crop `flattened_proposal_features_maps`.
    #   shape = [batch*max_num_proposals, crop_size, crop_size, feature_depth].

    batch, max_num_proposals, _ = utils.get_tensor_shape(proposals)
    box_ind = tf.expand_dims(tf.range(batch), axis=-1)
    box_ind = tf.tile(box_ind, [1, max_num_proposals])

    cropped_regions = tf.image.crop_and_resize(
        features_to_crop,
        boxes=tf.reshape(proposals, [-1, 4]),
        box_ind=tf.reshape(box_ind, [-1]),
        crop_size=[options.initial_crop_size, options.initial_crop_size])

    flattened_proposal_features_maps = slim.max_pool2d(
        cropped_regions,
        [options.maxpool_kernel_size, options.maxpool_kernel_size],
        stride=options.maxpool_stride)

    # Extract `proposal_features`,
    #   shape = [batch, max_num_proposals, feature_dims].

    (box_classifier_features
    ) = self._feature_extractor.extract_box_classifier_features(
        flattened_proposal_features_maps,
        scope='second_stage_feature_extraction')

    flattened_roi_pooled_features = tf.reduce_mean(
        box_classifier_features, [1, 2], name='AvgPool')
    flattened_roi_pooled_features = slim.dropout(
        flattened_roi_pooled_features,
        keep_prob=options.dropout_keep_prob,
        is_training=is_training)

    proposal_features = tf.reshape(flattened_roi_pooled_features,
                                   [batch, max_num_proposals, -1])

    # Assign weights from pre-trained checkpoint.

    tf.train.init_from_checkpoint(
        options.checkpoint_path,
        assignment_map={"/": "first_stage_feature_extraction/"})
    tf.train.init_from_checkpoint(
        options.checkpoint_path,
        assignment_map={"/": "second_stage_feature_extraction/"})

    return proposal_features

  def _build_midn_network(self,
                          num_proposals,
                          proposal_features,
                          num_classes=20):
    """Builds the Multiple Instance Detection Network.

    MIDN: An attention network.

    Args:
      num_proposals: A [batch] int tensor.
      proposal_features: A [batch, max_num_proposals, features_dims] 
        float tensor.
      num_classes: Number of classes.

    Returns:
      logits: A [batch, num_classes] float tensor.
      proba_r_given_c: A [batch, max_num_proposals, num_classes] float tensor.
    """
    with tf.name_scope('multi_instance_detection'):

      batch, max_num_proposals, _ = utils.get_tensor_shape(proposal_features)
      mask = tf.sequence_mask(
          num_proposals, maxlen=max_num_proposals, dtype=tf.float32)
      mask = tf.expand_dims(mask, axis=-1)

      # Calculates the values of following tensors:
      #   logits_r_given_c shape = [batch, max_num_proposals, num_classes].
      #   logits_c_given_r shape = [batch, max_num_proposals, num_classes].

      with tf.variable_scope('midn'):
        logits_r_given_c = slim.fully_connected(
            proposal_features,
            num_outputs=num_classes,
            activation_fn=None,
            scope='proba_r_given_c')
        logits_c_given_r = slim.fully_connected(
            proposal_features,
            num_outputs=num_classes,
            activation_fn=None,
            scope='proba_c_given_r')

      # Calculates the detection scores.

      proba_r_given_c = utils.masked_softmax(
          data=tf.multiply(mask, logits_r_given_c), mask=mask, dim=1)
      proba_r_given_c = tf.multiply(mask, proba_r_given_c)

      proba_c_given_r = tf.nn.softmax(logits_c_given_r)
      proba_c_given_r = tf.multiply(mask, proba_c_given_r)

      # Aggregates the logits.

      proposal_scores = tf.multiply(proba_c_given_r, proba_r_given_c)
      class_scores = tf.reduce_sum(proposal_scores, axis=1)

      tf.summary.histogram('midn/logits_r_given_c', logits_r_given_c)
      tf.summary.histogram('midn/logits_c_given_r', logits_c_given_r)
      tf.summary.histogram('midn/proposal_scores', proposal_scores)
      tf.summary.histogram('midn/class_scores', class_scores)

    return class_scores, proposal_scores

  def _build_latent_network(self,
                            num_proposals,
                            proposal_features,
                            num_classes=20,
                            num_latent_factors=20,
                            proba_h_given_c=None):
    """Builds the Multiple Instance Detection Network.

    MIDN: An attention network.

    Args:
      num_proposals: A [batch] int tensor.
      proposal_features: A [batch, max_num_proposals, features_dims] 
        float tensor.
      num_classes: Number of classes.
      proba_h_given_c: A [num_latent_factors, num_classes] float tensor.

    Returns:
      logits: A [batch, num_classes] float tensor.
      proba_r_given_c: A [batch, max_num_proposals, num_classes] float tensor.
      proba_h_given_c: A [num_latent_factors, num_classes] float tensor.
    """
    if proba_h_given_c is not None:
      assert proba_h_given_c.get_shape()[0].value == num_latent_factors

    with tf.name_scope('multi_instance_detection'):

      batch, max_num_proposals, _ = utils.get_tensor_shape(proposal_features)
      mask = tf.sequence_mask(
          num_proposals, maxlen=max_num_proposals, dtype=tf.float32)
      mask = tf.expand_dims(mask, axis=-1)

      # Calculates the values of following tensors:
      #   logits_c_given_r shape = [batch, max_num_proposals, num_classes].
      #   logits_r_given_h shape = [batch, max_num_proposals, num_hiddens].
      #   logits_h_given_c shape = [num_latent_factors, num_classes].

      with tf.variable_scope('midn'):
        logits_c_given_r = slim.fully_connected(
            proposal_features,
            num_outputs=num_classes,
            activation_fn=None,
            scope='proba_c_given_r')
        logits_r_given_h = slim.fully_connected(
            proposal_features,
            num_outputs=num_latent_factors,
            activation_fn=None,
            scope='proba_r_given_h')

        if proba_h_given_c is None:
          logits_h_given_c = slim.fully_connected(
              tf.diag(tf.ones([num_classes])),
              num_outputs=num_latent_factors,
              activation_fn=None,
              scope='proba_h_given_c')
          logits_h_given_c = tf.transpose(logits_h_given_c)
          proba_h_given_c = tf.nn.softmax(logits_h_given_c, axis=0)
          tf.summary.histogram('midn/logits_h_given_c', logits_h_given_c)

      # Marginalize `h` to get proba_r_given_c.

      logits_r_given_c = tf.matmul(
          tf.reshape(logits_r_given_h, [-1, num_latent_factors]),
          proba_h_given_c)
      logits_r_given_c = tf.reshape(logits_r_given_c,
                                    [batch, max_num_proposals, num_classes])

      proba_r_given_c = utils.masked_softmax(
          data=logits_r_given_c, mask=mask, dim=1)
      proba_r_given_c = tf.multiply(mask, proba_r_given_c)

      # Aggregates the logits.

      logits = tf.multiply(logits_c_given_r, proba_r_given_c)
      logits = tf.reduce_sum(logits, axis=1)

      tf.summary.histogram('midn/logits', logits)
      tf.summary.histogram('midn/logits_c_given_r', logits_c_given_r)
      tf.summary.histogram('midn/logits_r_given_h', logits_r_given_h)

    return logits, proba_r_given_c, proba_h_given_c

  def _post_process(self, inputs, predictions):
    """Post processes the predictions.

    Args:
      predictions: A dict mapping from name to tensor.

    Returns:
      predictions: A dict mapping from name to tensor.
    """
    options = self._model_proto

    results = {}

    # Post process to get the final detections.

    proposals = predictions[DetectionResultFields.proposal_boxes]

    for i in range(1 + options.oicr_iterations):
      post_process_fn = self._midn_post_process_fn
      proposal_scores = predictions[NODPredictions.oicr_proposal_scores +
                                    '_at_{}'.format(i)]
      if i > 0:
        post_process_fn = self._oicr_post_process_fn
        proposal_scores = tf.nn.softmax(proposal_scores, axis=-1)[:, :, 1:]

      # Post process.

      (num_detections, detection_boxes, detection_scores,
       detection_classes) = post_process_fn(proposals, proposal_scores)

      model_utils.visl_proposals_top_k(
          inputs,
          num_detections,
          detection_boxes,
          detection_scores,
          tf.gather(self._vocabulary_list, tf.to_int32(detection_classes - 1)),
          threshold=0.01,
          name='detection_{}'.format(i))

      results[DetectionResultFields.num_detections +
              '_at_{}'.format(i)] = num_detections
      results[DetectionResultFields.detection_boxes +
              '_at_{}'.format(i)] = detection_boxes
      results[DetectionResultFields.detection_scores +
              '_at_{}'.format(i)] = detection_scores
      results[DetectionResultFields.detection_classes +
              '_at_{}'.format(i)] = detection_classes
    return results

  def _build_prediction(self, examples, post_process=True):
    """Builds tf graph for prediction.

    Args:
      examples: dict of input tensors keyed by name.
      prediction_task: the specific prediction task.

    Returns:
      predictions: dict of prediction results keyed by name.
    """
    options = self._model_proto
    is_training = self._is_training

    (inputs, num_proposals,
     proposals) = (examples[InputDataFields.image],
                   examples[InputDataFields.num_proposals],
                   examples[InputDataFields.proposals])

    tf.summary.image('inputs', inputs, max_outputs=10)
    model_utils.visl_proposals(
        inputs, num_proposals, proposals, name='proposals', top_k=100)

    # FRCNN.

    proposal_features = self._extract_frcnn_feature(inputs, num_proposals,
                                                    proposals)

    # Build MIDN network.
    #   proba_r_given_c shape = [batch, max_num_proposals, num_classes].

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      if options.attention_type == nod_model_pb2.NODModel.PER_CLASS:
        midn_class_scores, midn_proposal_scores = self._build_midn_network(
            num_proposals, proposal_features, num_classes=self._num_classes)
      else:
        raise ValueError('Invalid attention type.')

    predictions = {
        DetectionResultFields.num_proposals: num_proposals,
        DetectionResultFields.proposal_boxes: proposals,
        NODPredictions.midn_class_scores: midn_class_scores,
    }

    # Build the OICR network.
    #   proposal_scores shape = [batch, max_num_proposals, 1 + num_classes].
    #   See `Multiple Instance Detection Network with OICR`.

    predictions[NODPredictions.oicr_proposal_scores +
                '_at_0'] = midn_proposal_scores

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      for i in range(options.oicr_iterations):
        predictions[NODPredictions.oicr_proposal_scores +
                    '_at_{}'.format(i + 1)] = slim.fully_connected(
                        proposal_features,
                        num_outputs=1 + self._num_classes,
                        activation_fn=None,
                        scope='oicr/iter{}'.format(i + 1))

    # Post process to get final predictions.

    if post_process:
      predictions.update(self._post_process(inputs, predictions))

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

    if is_training:
      return self._build_prediction(examples)

    return self._build_prediction(examples)

    #inputs = examples[InputDataFields.image]
    #assert inputs.get_shape()[0].value == 1

    #proposal_scores_list = [[] for _ in range(1 + options.oicr_iterations)]

    ## Get predictions from different resolutions.

    #for max_dimension in options.eval_max_dimension:
    #  inputs_resized = tf.expand_dims(
    #      imgproc.resize_image_to_max_dimension(inputs[0], max_dimension)[0],
    #      axis=0)
    #  examples[InputDataFields.image] = inputs_resized
    #  predictions = self._build_prediction(examples, post_process=False)

    #  for i in range(1 + options.oicr_iterations):
    #    proposals_scores = predictions[NODPredictions.oicr_proposal_scores +
    #                                   '_at_{}'.format(i)]
    #    proposal_scores_list[i].append(proposals_scores)

    #  tf.get_variable_scope().reuse_variables()

    ## Aggregate (averaging) predictions from different resolutions.

    #predictions_aggregated = predictions
    #for i in range(1 + options.oicr_iterations):
    #  proposal_scores = tf.stack(proposal_scores_list[i], axis=-1)
    #  proposal_scores = tf.reduce_max(proposal_scores, axis=-1)
    #  predictions_aggregated[NODPredictions.oicr_proposal_scores +
    #                         '_at_{}'.format(i)] = proposal_scores

    #predictions_aggregated.update(
    #    self._post_process(inputs, predictions_aggregated))

    #return predictions_aggregated

  def build_loss(self, predictions, examples, **kwargs):
    """Build tf graph to compute loss.

    Args:
      predictions: dict of prediction results keyed by name.
      examples: dict of inputs keyed by name.

    Returns:
      loss_dict: dict of loss tensors keyed by name.
    """
    options = self._model_proto

    loss_dict = {}

    with tf.name_scope('losses'):

      # Extract image-level labels.

      labels = self._extract_class_label(
          class_texts=examples[InputDataFields.caption_strings],
          vocabulary_list=self._vocabulary_list)

      # Loss of the multi-instance detection network.

      midn_class_scores = tf.clip_by_value(
          predictions[NODPredictions.midn_class_scores], _EPSILON, 1 - _EPSILON)
      losses = -labels * tf.log(midn_class_scores) - (
          1.0 - labels) * tf.log(1.0 - midn_class_scores)
      loss_dict['midn_cross_entropy_loss'] = tf.reduce_mean(losses)

      # Losses of the online instance classifier refinement network.

      (num_proposals, proposals, proposal_scores_0) = (
          predictions[DetectionResultFields.num_proposals],
          predictions[DetectionResultFields.proposal_boxes],
          predictions[NODPredictions.oicr_proposal_scores + '_at_0'])

      batch, max_num_proposals, _ = utils.get_tensor_shape(proposal_scores_0)
      proposal_scores_0 = tf.concat(
          [tf.fill([batch, max_num_proposals, 1], 0.0), proposal_scores_0],
          axis=-1)

      global_step = tf.train.get_or_create_global_step()
      oicr_loss_mask = tf.cast(global_step > options.oicr_start_step,
                               tf.float32)

      for i in range(options.oicr_iterations):
        proposal_scores_1 = predictions[NODPredictions.oicr_proposal_scores +
                                        '_at_{}'.format(i + 1)]
        oicr_cross_entropy_loss_at_i = model_utils.calc_oicr_loss(
            labels,
            num_proposals,
            proposals,
            proposal_scores_0,
            proposal_scores_1,
            scope='oicr_{}'.format(i + 1),
            iou_threshold=options.oicr_iou_threshold)
        loss_dict['oicr_cross_entropy_loss_at_{}'.format(
            i + 1)] = oicr_loss_mask * oicr_cross_entropy_loss_at_i

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
