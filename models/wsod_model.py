from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from models.model_base import ModelBase
from protos import nod2_model_pb2

from nets import nets_factory
from nets import vgg
from core import imgproc
from core import utils
from core import plotlib
from core.standard_fields import InputDataFields
from core.standard_fields import NOD2Predictions
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
  """NOD2 model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of nod2_model_pb2.NOD2Model
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, nod2_model_pb2.NOD2Model):
      raise ValueError('The model_proto has to be an instance of NOD2Model.')

    options = model_proto

    self._vocabulary_list = model_utils.read_vocabulary(options.vocabulary_file)

    self._num_classes = len(self._vocabulary_list)

    self._feature_extractor = build_faster_rcnn_feature_extractor(
        options.feature_extractor, is_training,
        options.inplace_batchnorm_update)

    self._pcl_preprocess_fn = function_builder.build_post_processor(
        options.pcl_preprocess)

    self._midn_post_process_fn = function_builder.build_post_processor(
        options.midn_post_process)

    self._oicr_post_process_fn = function_builder.build_post_processor(
        options.oicr_post_process)

  def _extract_class_label(self, class_texts, vocabulary_list):
    """Extracts class labels.

    Args:
      class_texts: a [batch, max_num_objects] string tensor.
      vocabulary_list: a list of words of length `num_classes`.

    Returns:
      labels: a [batch, num_classes] float tensor.
    """
    with tf.name_scope('extract_class_label'):
      batch, _ = utils.get_tensor_shape(class_texts)

      categorical_col = tf.feature_column.categorical_column_with_vocabulary_list(
          key='name_to_id', vocabulary_list=vocabulary_list, num_oov_buckets=1)
      indicator_col = tf.feature_column.indicator_column(categorical_col)
      indicator = tf.feature_column.input_layer({'name_to_id': class_texts},
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

  def _build_midn_network_tanh(self,
                               num_proposals,
                               proposal_features,
                               num_classes=20,
                               tanh_hiddens=50):
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
        tanh_output = slim.fully_connected(
            proposal_features,
            num_outputs=tanh_hiddens,
            activation_fn=tf.nn.tanh,
            scope='tanh_output')
        logits_r_given_c = slim.fully_connected(
            tanh_output,
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

      # Aggregates the logits.

      class_logits = tf.multiply(logits_c_given_r, proba_r_given_c)
      class_logits = utils.masked_sum(data=class_logits, mask=mask, dim=1)

      proposal_scores = tf.multiply(
          tf.nn.sigmoid(class_logits), proba_r_given_c)
      #proposal_scores = tf.multiply(
      #    tf.nn.softmax(class_logits), proba_r_given_c)

      tf.summary.histogram('midn/logits_r_given_c', logits_r_given_c)
      tf.summary.histogram('midn/logits_c_given_r', logits_c_given_r)
      tf.summary.histogram('midn/proposal_scores', proposal_scores)
      tf.summary.histogram('midn/class_logits', class_logits)

    return tf.squeeze(class_logits, axis=1), proposal_scores, proba_r_given_c

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

      # Aggregates the logits.

      class_logits = tf.multiply(logits_c_given_r, proba_r_given_c)
      class_logits = utils.masked_sum(data=class_logits, mask=mask, dim=1)

      proposal_scores = tf.multiply(
          tf.nn.sigmoid(class_logits), proba_r_given_c)
      #proposal_scores = tf.multiply(
      #    tf.nn.softmax(class_logits), proba_r_given_c)

      tf.summary.histogram('midn/logits_r_given_c', logits_r_given_c)
      tf.summary.histogram('midn/logits_c_given_r', logits_c_given_r)
      tf.summary.histogram('midn/proposal_scores', proposal_scores)
      tf.summary.histogram('midn/class_logits', class_logits)

    return tf.squeeze(class_logits, axis=1), proposal_scores, proba_r_given_c

  def _build_latent_network(self,
                            num_proposals,
                            proposal_features,
                            num_classes=20,
                            num_latent_factors=20,
                            proba_h_given_c=None,
                            proba_h_use_sigmoid=False):
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

    batch, max_num_proposals, _ = utils.get_tensor_shape(proposal_features)
    mask = tf.sequence_mask(
        num_proposals, maxlen=max_num_proposals, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=-1)

    # Calculates the values of following tensors:
    #   logits_c_given_r shape = [batch, max_num_proposals, num_classes].
    #   logits_r_given_h shape = [batch, max_num_proposals, num_latent_factors].
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
        tf.summary.histogram('logits_h_given_c', logits_h_given_c)

        if proba_h_use_sigmoid:
          proba_h_given_c = tf.nn.sigmoid(logits_h_given_c)
        else:
          proba_h_given_c = tf.nn.softmax(logits_h_given_c, axis=0)

    logits_r_given_c = tf.matmul(
        tf.reshape(logits_r_given_h, [-1, num_latent_factors]), proba_h_given_c)
    logits_r_given_c = tf.reshape(logits_r_given_c,
                                  [batch, max_num_proposals, num_classes])

    proba_r_given_c = utils.masked_softmax(
        data=logits_r_given_c, mask=mask, dim=1)
    proba_r_given_c = tf.multiply(mask, proba_r_given_c)

    # Aggregates the logits.

    class_logits = tf.multiply(logits_c_given_r, proba_r_given_c)
    class_logits = utils.masked_sum(data=class_logits, mask=mask, dim=1)

    proposal_scores = tf.multiply(tf.nn.sigmoid(class_logits), proba_r_given_c)
    #proposal_scores = tf.multiply(
    #    tf.nn.softmax(class_logits), proba_r_given_c)

    tf.summary.histogram('midn/logits_c_given_r', logits_c_given_r)
    tf.summary.histogram('midn/logits_r_given_h', logits_r_given_h)
    tf.summary.histogram('midn/class_logits', class_logits)

    return (tf.squeeze(class_logits, axis=1), proposal_scores, proba_r_given_c,
            proba_h_given_c)

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
      proposal_scores = predictions[NOD2Predictions.oicr_proposal_scores +
                                    '_at_{}'.format(i)]
      proposal_scores = tf.stop_gradient(proposal_scores)
      if i > 0:
        post_process_fn = self._oicr_post_process_fn
        proposal_scores = tf.nn.softmax(proposal_scores, axis=-1)[:, :, 1:]

      # Post process.

      (num_detections, detection_boxes, detection_scores, detection_classes,
       _) = post_process_fn(proposals, proposal_scores)

      model_utils.visl_detections(
          inputs,
          num_detections,
          detection_boxes,
          detection_scores,
          tf.gather(self._vocabulary_list, tf.to_int32(detection_classes - 1)),
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

    # Build the OICR network.
    #   proposal_scores shape = [batch, max_num_proposals, 1 + num_classes].
    #   See `Multiple Instance Detection Network with OICR`.

    predictions = {}
    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      for i in range(options.oicr_iterations):
        predictions[NOD2Predictions.oicr_proposal_scores + '_at_{}'.format(
            i + 1)] = proposal_scores = slim.fully_connected(
                proposal_features,
                num_outputs=1 + self._num_classes,
                activation_fn=None,
                scope='oicr/iter{}'.format(i + 1))

    if post_process and options.HasField('pcl_preprocess'):
      proposal_scores = tf.nn.softmax(
          tf.stop_gradient(proposal_scores), axis=-1)[:, :, 1:]
      (num_proposals, proposals,
       _, _, additional_fields) = self._pcl_preprocess_fn(
           proposals, proposal_scores, {'proposal_features': proposal_features})
      proposal_features = additional_fields['proposal_features']

    # Build MIDN network.
    #   proba_r_given_c shape = [batch, max_num_proposals, num_classes].

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      if options.attention_type == nod2_model_pb2.NOD2Model.PER_CLASS:
        (midn_class_logits, midn_proposal_scores,
         midn_proba_r_given_c) = self._build_midn_network(
             num_proposals, proposal_features, num_classes=self._num_classes)
      elif options.attention_type == nod2_model_pb2.NOD2Model.PER_CLASS_TANH:
        (midn_class_logits, midn_proposal_scores,
         midn_proba_r_given_c) = self._build_midn_network_tanh(
             num_proposals, proposal_features, num_classes=self._num_classes)
      else:
        raise ValueError('Invalid attention type.')

    predictions.update({
        DetectionResultFields.class_labels:
        tf.constant(self._vocabulary_list),
        DetectionResultFields.num_proposals:
        num_proposals,
        DetectionResultFields.proposal_boxes:
        proposals,
        NOD2Predictions.midn_class_logits:
        midn_class_logits,
        NOD2Predictions.midn_proba_r_given_c:
        midn_proba_r_given_c,
        NOD2Predictions.oicr_proposal_scores + '_at_0':
        midn_proposal_scores
    })

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

    if is_training or len(options.eval_min_dimension) == 0:
      return self._build_prediction(examples)

    inputs = examples[InputDataFields.image]
    assert inputs.get_shape()[0].value == 1

    proposal_scores_list = [[] for _ in range(1 + options.oicr_iterations)]

    # Get predictions from different resolutions.

    reuse = False
    for min_dimension in options.eval_min_dimension:
      inputs_resized = tf.expand_dims(
          imgproc.resize_image_to_min_dimension(inputs[0], min_dimension)[0],
          axis=0)
      examples[InputDataFields.image] = inputs_resized

      with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        predictions = self._build_prediction(examples, post_process=False)

      for i in range(1 + options.oicr_iterations):
        proposals_scores = predictions[NOD2Predictions.oicr_proposal_scores +
                                       '_at_{}'.format(i)]
        proposal_scores_list[i].append(proposals_scores)

      reuse = True

    # Aggregate (averaging) predictions from different resolutions.

    predictions_aggregated = predictions
    for i in range(1 + options.oicr_iterations):
      proposal_scores = tf.stack(proposal_scores_list[i], axis=-1)
      proposal_scores = tf.reduce_mean(proposal_scores, axis=-1)
      predictions_aggregated[NOD2Predictions.oicr_proposal_scores +
                             '_at_{}'.format(i)] = proposal_scores

    predictions_aggregated.update(
        self._post_process(inputs, predictions_aggregated))

    return predictions_aggregated

  def _midn_loss_mine_hardest_negative(self, labels, losses):
    """Hardest negative mining of the MIDN loss.

    Args:
      labels: A [batch, num_classes] float tensor, where `1` denotes the 
        presence of a class.
      losses: A [batch, num_classes] float tensor, the losses predicted by  
        the model.

    Returns:
      mask: A [batch, num_classes] float tensor where `1` denotes the 
        selected entry.
    """
    batch, num_classes = utils.get_tensor_shape(labels)
    indices_0 = tf.range(batch, dtype=tf.int64)
    indices_1 = utils.masked_argmax(data=losses, mask=1.0 - labels, dim=1)
    indices = tf.stack([indices_0, indices_1], axis=-1)
    negative_masks = tf.sparse_to_dense(
        indices, [batch, num_classes], sparse_values=1.0)
    return tf.add(labels, negative_masks)

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

      if not options.caption_as_label:
        labels = self._extract_class_label(
            class_texts=examples[InputDataFields.object_texts],
            vocabulary_list=self._vocabulary_list)
      else:
        labels = self._extract_class_label(
            class_texts=slim.flatten(examples[InputDataFields.caption_strings]),
            vocabulary_list=self._vocabulary_list)

      # A prediction model from caption to class

      # Loss of the multi-instance detection network.

      midn_class_logits = predictions[NOD2Predictions.midn_class_logits]
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=midn_class_logits)

      # Hard-negative mining.

      if options.midn_loss_negative_mining == nod2_model_pb2.NOD2Model.NONE:
        if options.classification_loss_use_sum:
          assert False
          loss_dict['midn_cross_entropy_loss'] = tf.multiply(
              tf.reduce_mean(tf.reduce_sum(losses, axis=-1)),
              options.midn_loss_weight)
        else:
          if options.caption_as_label:
            loss_masks = tf.to_float(tf.reduce_any(labels > 0, axis=-1))
            loss_dict['midn_cross_entropy_loss'] = tf.multiply(
                tf.squeeze(
                    utils.masked_avg(
                        tf.reduce_mean(losses, axis=-1), mask=loss_masks,
                        dim=0)), options.midn_loss_weight)
          else:
            loss_dict['midn_cross_entropy_loss'] = tf.multiply(
                tf.reduce_mean(losses), options.midn_loss_weight)
      elif options.midn_loss_negative_mining == nod2_model_pb2.NOD2Model.HARDEST:
        assert False
        loss_masks = self._midn_loss_mine_hardest_negative(labels, losses)
        loss_dict['midn_cross_entropy_loss'] = tf.reduce_mean(
            utils.masked_avg(data=losses, mask=loss_masks, dim=1))
      else:
        raise ValueError('Invalid negative mining method.')

      # Losses of the online instance classifier refinement network.

      (num_proposals,
       proposals) = (predictions[DetectionResultFields.num_proposals],
                     predictions[DetectionResultFields.proposal_boxes])
      batch, max_num_proposals, _ = utils.get_tensor_shape(proposals)

      proposal_scores_0 = predictions[NOD2Predictions.oicr_proposal_scores +
                                      '_at_0']
      if options.oicr_use_proba_r_given_c:
        proposal_scores_0 = predictions[NOD2Predictions.midn_proba_r_given_c]

      proposal_scores_0 = tf.concat(
          [tf.fill([batch, max_num_proposals, 1], 0.0), proposal_scores_0],
          axis=-1)

      global_step = tf.train.get_or_create_global_step()
      oicr_loss_mask = tf.cast(global_step > options.oicr_start_step,
                               tf.float32)

      for i in range(options.oicr_iterations):
        proposal_scores_1 = predictions[NOD2Predictions.oicr_proposal_scores +
                                        '_at_{}'.format(i + 1)]
        oicr_cross_entropy_loss_at_i = model_utils.calc_oicr_loss(
            labels,
            num_proposals,
            proposals,
            tf.stop_gradient(proposal_scores_0),
            proposal_scores_1,
            scope='oicr_{}'.format(i + 1),
            iou_threshold=options.oicr_iou_threshold)
        loss_dict['oicr_cross_entropy_loss_at_{}'.format(i + 1)] = tf.multiply(
            oicr_loss_mask * oicr_cross_entropy_loss_at_i,
            options.oicr_loss_weight)

        proposal_scores_0 = tf.nn.softmax(proposal_scores_1, axis=-1)

      # Min-entropy loss.

      mask = tf.sequence_mask(
          num_proposals, maxlen=max_num_proposals, dtype=tf.float32)
      proba_r_given_c = predictions[NOD2Predictions.midn_proba_r_given_c]
      losses = tf.log(proba_r_given_c + _EPSILON)
      losses = tf.squeeze(
          utils.masked_sum_nd(data=losses, mask=mask, dim=1), axis=1)
      min_entropy_loss = tf.reduce_mean(tf.reduce_sum(losses * labels, axis=1))
      min_entropy_loss = tf.multiply(min_entropy_loss,
                                     options.min_entropy_loss_weight)

      max_proba = tf.reduce_mean(
          utils.masked_maximum(
              data=proba_r_given_c, mask=tf.expand_dims(mask, -1), dim=1))
      tf.losses.add_loss(min_entropy_loss)

    tf.summary.scalar('loss/min_entropy_loss', min_entropy_loss)
    tf.summary.scalar('loss/max_proba', max_proba)

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
