from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from models.model_base import ModelBase
from protos import stacked_attn_model_pb2

from nets import nets_factory
from nets import vgg
from core import imgproc
from core import utils
from core import plotlib
from core.standard_fields import InputDataFields
from core.standard_fields import StackedAttnPredictions
from core.standard_fields import DetectionResultFields
from core.training_utils import build_hyperparams
from core import init_grid_anchors
from models import utils as model_utils
from core import box_utils

from object_detection.builders import hyperparams_builder
from object_detection.builders.model_builder import _build_faster_rcnn_feature_extractor as build_faster_rcnn_feature_extractor

slim = tf.contrib.slim


class Model(ModelBase):
  """NOD model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of stacked_attn_model_pb2.StackedAttnModel
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, stacked_attn_model_pb2.StackedAttnModel):
      raise ValueError(
          'The model_proto has to be an instance of StackedAttnModel.')

    options = model_proto

    self._vocabulary_list = model_utils.read_vocabulary(options.vocabulary_file)

    self._num_classes = len(self._vocabulary_list)

    self._feature_extractor = build_faster_rcnn_feature_extractor(
        options.feature_extractor, is_training,
        options.inplace_batchnorm_update)

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

    (inputs, num_proposals,
     proposals) = (examples[InputDataFields.image],
                   examples[InputDataFields.num_proposals],
                   examples[InputDataFields.proposals])

    predictions = {
        DetectionResultFields.num_proposals: num_proposals,
        DetectionResultFields.proposal_boxes: proposals,
    }

    # FRCNN.
    #   `proposal_features` shape = [batch, max_num_proposals, feature_dims].
    #   `proposal_masks` shape = [batch, max_num_proposals].

    proposal_features = self._extract_frcnn_feature(inputs, num_proposals,
                                                    proposals)

    batch, max_num_proposals, _ = utils.get_tensor_shape(proposal_features)
    proposal_masks = tf.sequence_mask(
        num_proposals, maxlen=max_num_proposals, dtype=tf.float32)

    # Build the SADDN predictions.
    #   `logits_c_given_r` shape = [batch, max_num_proposals, num_classes].
    #   `logits_r_given_c` shape = [batch, max_num_proposals, num_classes].

    with tf.variable_scope('SADDN'), \
        slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):

      logits_c_given_r = slim.fully_connected(
          proposal_features,
          num_outputs=self._num_classes,
          activation_fn=None,
          scope='proba_c_given_r')
      logits_r_given_c = slim.fully_connected(
          proposal_features,
          num_outputs=self._num_classes,
          activation_fn=None,
          scope='proba_r_given_c')

      proba_c_given_r = tf.nn.softmax(logits_c_given_r)
      proba_r_given_c = utils.masked_softmax(
          data=logits_r_given_c,
          mask=tf.expand_dims(proposal_masks, axis=-1),
          dim=1)
      proba_r_given_c = tf.multiply(
          tf.expand_dims(proposal_masks, axis=-1), proba_r_given_c)

    tf.summary.image('inputs', inputs, max_outputs=10)
    model_utils.visl_proposals(
        inputs, num_proposals, proposals, name='proposals', top_k=2000)

    # SADDN iterations.

    logits_at_0 = utils.masked_avg_nd(
        data=logits_c_given_r, mask=proposal_masks, dim=1)
    logits_at_0 = tf.squeeze(logits_at_0, axis=1)

    logits_at_i = logits_at_0
    for i in range(options.saddn_iterations):
      # Infer the proba_r_given_c.

      # Infer the proba_c.

      proba_c_at_i = tf.nn.softmax(logits_at_i)
      import pdb
      pdb.set_trace()

      proba_r_at_i = tf.multiply(
          tf.expand_dims(proba_c_at_i, axis=1), proba_r_given_c)
      proba_r_at_i = tf.reduce_sum(proba_r_at_i, axis=-1, keepdims=True)

      # Infer the detection results at iter `i`.

      (num_detections_at_i, detection_boxes_at_i, detection_scores_at_i,
       detection_classes_at_i) = model_utils.post_process(
           proposals, proba_r_at_i * proba_c_given_r)

      (predictions[StackedAttnPredictions.logits + '_at_{}'.format(i)],
       predictions[DetectionResultFields.num_detections + '_at_{}'.format(i)],
       predictions[DetectionResultFields.detection_boxes + '_at_{}'.format(i)],
       predictions[DetectionResultFields.detection_scores + '_at_{}'.format(i)],
       predictions[DetectionResultFields.detection_classes +
                   '_at_{}'.format(i)]) = (logits_at_i, num_detections_at_i,
                                           detection_boxes_at_i,
                                           detection_scores_at_i,
                                           detection_classes_at_i)

      model_utils.visl_proposals_top_k(
          inputs,
          num_detections_at_i,
          detection_boxes_at_i,
          detection_scores_at_i,
          tf.gather(self._vocabulary_list,
                    tf.to_int32(detection_classes_at_i - 1)),
          name='detection_{}'.format(i))

      # `logits_at_i` for the next iteration.

      logits_at_i = tf.multiply(proba_r_at_i, logits_c_given_r)
      logits_at_i = tf.reduce_sum(logits_at_i, axis=1)

    return predictions

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

      # Extract and normalize image-level labels.

      labels = self._extract_class_label(
          class_texts=examples[InputDataFields.caption_strings],
          vocabulary_list=self._vocabulary_list)
      labels = tf.div(labels, tf.reduce_sum(labels, axis=-1, keepdims=True))

      for i in range(options.saddn_iterations):
        logits = predictions[StackedAttnPredictions.logits + '_at_{}'.format(i)]
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        loss_dict['cross_entropy_loss_at_{}'.format(i)] = tf.reduce_mean(losses)

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
