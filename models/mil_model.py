from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from models.model_base import ModelBase
from protos import mil_model_pb2

from nets import nets_factory
from core import imgproc
from core import utils
from core import plotlib
from core.standard_fields import InputDataFields
from core.standard_fields import MILTasks
from core.standard_fields import MILPredictions
from core.standard_fields import MILVariableScopes
from core.standard_fields import DetectionResultFields
from core.training_utils import build_hyperparams
from core import init_grid_anchors
from models import utils as model_utils

from object_detection.core.post_processing import batch_multiclass_non_max_suppression

slim = tf.contrib.slim


class Model(ModelBase):
  """MIL model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of mil_model_pb2.MILModel
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, mil_model_pb2.MILModel):
      raise ValueError('The model_proto has to be an instance of MILModel.')

    self._vocabulary_list = model_utils.read_vocabulary(
        model_proto.vocabulary_file)
    self._num_classes = len(self._vocabulary_list)

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      a list of model variables or None by default.
    """
    options = self._model_proto

    # Filter out CNN variables that are not trainable.
    variables_to_train = []
    for var in tf.trainable_variables():
      if not options.cnn.trainable:
        if options.cnn.scope in var.op.name:
          tf.logging.info("Freeze cnn parameter %s.", var.op.name)
          continue
      variables_to_train.append(var)

    for var in variables_to_train:
      tf.logging.info("Model variables: %s.", var.op.name)
    return variables_to_train

  def _visl_proposals(self,
                      image,
                      num_proposals,
                      proposals,
                      top_k=100,
                      height=224,
                      width=224):
    """Visualize proposal results to the tensorboard.

    Args:
      image: A [batch, height, width, channels] float tensor, 
        ranging from 0 to 255.
      num_proposals: A [batch] int tensor.
      proposals: A [batch, max_num_proposals, 4] float tensor.
      height: Height of the visualized image.
      width: Width of the visualized image.
    """
    with tf.name_scope('visl_proposals'):
      image = tf.image.resize_images(image, [height, width])
      image = tf.cast(image, tf.uint8)
      image = plotlib.draw_rectangles(
          image, boxes=proposals[:, :top_k, :], color=plotlib.RED)
    tf.summary.image("proposals", image, max_outputs=5)

  def _visl_proposals_top_k(self,
                            image,
                            num_proposals,
                            proposals,
                            proposal_scores,
                            top_k=5,
                            height=224,
                            width=224):
    """Visualize top proposal results to the tensorboard.

    Args:
      image: A [batch, height, width, channels] float tensor, 
        ranging from 0 to 255.
      num_proposals: A [batch] int tensor.
      proposals: A [batch, max_num_proposals, 4] float tensor.
      proposal_scores: A [batch, max_num_proposals, num_classes] float tensor.
      height: Height of the visualized image.
      width: Width of the visualized image.
    """
    with tf.name_scope('visl_proposals'):
      image = tf.image.resize_images(image, [height, width])
      image = tf.cast(image, tf.uint8)

      top_k_boxes, top_k_scores, _ = model_utils.get_top_k_boxes_and_scores(
          proposals, tf.reduce_max(proposal_scores, axis=-1), k=top_k)
      image = plotlib.draw_rectangles(
          image, boxes=top_k_boxes, scores=top_k_scores, color=plotlib.RED)
    tf.summary.image("proposals", image, max_outputs=5)

  def _calc_spp_feature(self, inputs, spp_bins=[1, 2, 3, 6]):
    """Apply SPP layer to get the multi-resolutional feature.

    LIMITATION: the inputs has to have static shape.

    Args:
      inputs: A [batch, feature_height, feature_width, feature_dims] 
        float tensor.
      spp_bins: A python list representing the number of bins at each SPP 
        level. 

    Returns:
      spp_pool: A [batch, spp_feature_dims] fixed-length feature tensor.

    Raises:
      ValueError: If any of the parameters are invalid.
    """
    batch, height, width, _ = utils.get_tensor_shape(inputs)
    if not type(height) == type(width) == int:
      raise ValueError('The inputs should have static shape.')

    with tf.name_scope('calc_spp_feature'):
      pool_outputs = []
      for bins in spp_bins:
        if height % bins or width % bins:
          raise ValueError('Reminder should be ZERO.')

        pool_h, pool_w = height // bins, width // bins
        stride_h, stride_w = height // bins, width // bins
        pool = tf.nn.max_pool(
            inputs,
            ksize=[1, pool_h, pool_w, 1],
            strides=[1, stride_h, stride_w, 1],
            padding='SAME')
        pool_outputs.append(tf.reshape(pool, [batch, -1]))
        tf.logging.info(
            'SPP bins=%i, bin_size=(%i,%i), strides=(%i, %i), output=%s', bins,
            pool_h, pool_w, stride_h, stride_w,
            pool.get_shape().as_list())
      spp_pool = tf.concat(pool_outputs, axis=-1)
      tf.logging.info('Final SPP shape=%s', spp_pool.get_shape().as_list())

    return spp_pool

  def _calc_spp_proposal_feature(self, image_feature_cropped):
    """Calculates proposal feature using spp.

    Args:
      image_feature_cropped: A [batch, crop_size, crop_size, feature_dims]
        float tensor.

    Returns:
      proposal_feature: A [batch, proposal_feature_dims] float tensor.
    """
    options = self._model_proto
    is_training = self._is_training

    spp_feature_cropped = self._calc_spp_feature(
        image_feature_cropped, spp_bins=[lv for lv in options.spp_bins])

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      net = spp_feature_cropped
      for i in range(options.hidden_layers):
        net = slim.fully_connected(
            net,
            num_outputs=options.hidden_units,
            scope='hidden/fc_{}'.format(i + 1))
        net = slim.dropout(
            net, options.hidden_dropout_keep_prob, is_training=is_training)
    return net

  def _calc_vgg_fc_proposal_feature(self, image_feature_cropped):
    """Calculates proposal feature using spp.

    Args:
      image_feature_cropped: A [batch, crop_size, crop_size, feature_dims]
        float tensor.

    Returns:
      proposal_feature: A [batch, proposal_feature_dims] float tensor.
    """
    options = self._model_proto
    is_training = self._is_training

    assert options.cnn.name == 'vgg_16'

    with tf.variable_scope(options.cnn.scope, reuse=True):
      with tf.variable_scope('vgg_16'):
        net = image_feature_cropped
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(
            net,
            options.cnn.dropout_keep_prob,
            is_training=is_training and options.cnn.trainable,
            scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(
            net,
            options.hidden_dropout_keep_prob,
            is_training=is_training,
            scope='dropout7')
        net = tf.reduce_mean(net, [1, 2], name='global_pool')
    return net

  def _build_midn_network(self, num_proposals, proposal_feature, num_classes=20):
    """Builds the Multiple Instance Detection Network.

    MIDN: An attention network.

    Args:
      num_proposals: A [batch] it tensor.
      proposal_feature: A [batch, max_num_proposals, feature_dims] 
        float tensor.
      num_classes: Number of classes.

    Returns:
      proposal_scores: A [batch, max_num_proposals, num_classes] float tensor.
    """
    with tf.name_scope('multi_instance_detection'):

      _, max_num_proposals, _ = utils.get_tensor_shape(proposal_feature)

      # branch1/branch2 shape = [batch, max_num_proposals, num_classes.]

      branch1 = slim.fully_connected(
          proposal_feature,
          num_outputs=num_classes,
          activation_fn=None,
          scope='midn/branch1')
      branch2 = slim.fully_connected(
          proposal_feature,
          num_outputs=num_classes,
          activation_fn=None,
          scope='midn/branch2')

      proba_c_given_r = tf.nn.softmax(branch1, axis=2)

      #proba_r_given_c = tf.nn.softmax(branch2, axis=1)
      mask = tf.sequence_mask(
          num_proposals, maxlen=max_num_proposals, dtype=tf.float32)
      mask = tf.expand_dims(mask, axis=-1)
      proba_r_given_c = utils.masked_softmax(data=branch2, mask=mask, dim=1)

      proposal_scores = tf.multiply(proba_c_given_r, proba_r_given_c)

    return proposal_scores

  def _post_process(self,
                    boxes,
                    scores,
                    score_thresh=-0.1,
                    iou_thresh=0.5,
                    max_size_per_class=100,
                    max_total_size=300):
    """Applies post process to get the final detections.

    Args:
      boxes: A [batch_size, num_anchors, q, 4] float32 tensor containing
        detections. If `q` is 1 then same boxes are used for all classes
          otherwise, if `q` is equal to number of classes, class-specific boxes
          are used.
      scores: A [batch_size, num_anchors, num_classes] float32 tensor containing
        the scores for each of the `num_anchors` detections. The scores have to be
        non-negative when use_static_shapes is set True.
      score_thresh: scalar threshold for score (low scoring boxes are removed).
      iou_thresh: scalar threshold for IOU (new boxes that have high IOU overlap
        with previously selected boxes are removed).
      max_size_per_class: maximum number of retained boxes per class.
      max_total_size: maximum number of boxes retained over all classes. By
        default returns all boxes retained after capping boxes per class.

  Returns:
    num_detections: A [batch_size] int32 tensor indicating the number of
      valid detections per batch item. Only the top num_detections[i] entries in
      nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
      entries are zero paddings.
    nmsed_boxes: A [batch_size, max_detections, 4] float32 tensor
      containing the non-max suppressed boxes.
    nmsed_scores: A [batch_size, max_detections] float32 tensor containing
      the scores for the boxes.
    nmsed_classes: A [batch_size, max_detections] float32 tensor
      containing the class for boxes.
    """
    boxes = tf.expand_dims(boxes, axis=2)
    (nmsed_boxes, nmsed_scores, nmsed_classes, _, _,
     num_detections) = batch_multiclass_non_max_suppression(
         boxes,
         scores,
         score_thresh=score_thresh,
         iou_thresh=iou_thresh,
         max_size_per_class=max_size_per_class,
         max_total_size=max_total_size)
    return num_detections, nmsed_boxes, nmsed_scores, nmsed_classes + 1

  def build_prediction(self,
                       examples,
                       prediction_task=MILTasks.image_label,
                       **kwargs):
    """Builds tf graph for prediction.

    Args:
      examples: dict of input tensors keyed by name.
      prediction_task: the specific prediction task.

    Returns:
      predictions: dict of prediction results keyed by name.
    """
    options = self._model_proto
    is_training = self._is_training

    (image, num_proposals,
     proposals) = (examples[InputDataFields.image],
                   examples[InputDataFields.num_proposals],
                   examples[InputDataFields.proposals])

    # Use the CNN to extract feature.
    #   image_feature shape=[batch, feature_height, feature_width, feature_dims]

    image_feature = model_utils.calc_cnn_feature(
        image, options.cnn, is_training=is_training)

    # Crop image feature from the CNN output.
    #   image_feature_cropped_and_flattened
    #   shape=[batch*max_num_proposals, crop_size, crop_size, feature_dims]

    batch, max_num_proposals, _ = utils.get_tensor_shape(proposals)
    box_ind = tf.expand_dims(tf.range(batch), axis=-1)
    box_ind = tf.tile(box_ind, [1, max_num_proposals])

    crop_size = options.feature_crop_size
    image_feature_cropped = tf.image.crop_and_resize(
        image_feature,
        boxes=tf.reshape(proposals, [-1, 4]),
        box_ind=tf.reshape(box_ind, [-1]),
        crop_size=[crop_size, crop_size],
        method='bilinear')

    # Get the multi-resolutional feature.
    #   proposal_feature shape=[batch, max_num_proposals, hidden_units].

    if options.feature_extractor == mil_model_pb2.MILModel.SPP:
      proposal_feature = self._calc_spp_proposal_feature(image_feature_cropped)
    else:
      proposal_feature = self._calc_vgg_fc_proposal_feature(
          image_feature_cropped)
    proposal_feature = tf.reshape(proposal_feature,
                                  [batch, max_num_proposals, -1])

    # Build the MIDN network.
    #   proposal_scores shape = [batch, max_num_proposals, num_classes].
    #   See `Multiple Instance Detection Network with OICR`.

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      proposal_scores = self._build_midn_network(
          num_proposals, proposal_feature, num_classes=self._num_classes)
      midn_logits = tf.reduce_sum(proposal_scores, axis=1)

    # Post process to get the final detections.

    (num_detections, detection_boxes, detection_scores,
     detection_classes) = self._post_process(proposals, proposal_scores)

    # Visualize the detection results.

    self._visl_proposals(image, num_proposals, proposals)
    self._visl_proposals_top_k(image, num_proposals, proposals, proposal_scores)

    return {
        MILPredictions.midn_logits: midn_logits,
        DetectionResultFields.num_detections: num_detections,
        DetectionResultFields.detection_boxes: detection_boxes,
        DetectionResultFields.detection_scores: detection_scores,
        DetectionResultFields.detection_classes: detection_classes
    }

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

  def build_loss(self, predictions, examples, **kwargs):
    """Build tf graph to compute loss.

    Args:
      predictions: dict of prediction results keyed by name.
      examples: dict of inputs keyed by name.

    Returns:
      loss_dict: dict of loss tensors keyed by name.
    """
    loss_dict = {}

    with tf.name_scope('losses'):

      labels = self._extract_class_label(
          class_texts=examples[InputDataFields.caption_strings],
          vocabulary_list=self._vocabulary_list)

      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=predictions[MILPredictions.midn_logits])
      loss_dict['midn_cross_entropy_loss'] = tf.reduce_mean(losses)

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
    metrics = {}

    with tf.name_scope('evaluation'):
      labels = self._extract_class_label(
          class_texts=examples[InputDataFields.caption_strings],
          vocabulary_list=self._vocabulary_list)

      for (label, logit) in zip(
          tf.unstack(labels, axis=0),
          tf.unstack(predictions[MILPredictions.midn_logits], axis=0)):
        for top_k in [1]:
          label_indices = tf.squeeze(tf.where(tf.greater(label, 0)), axis=-1)
          map_val, map_update = tf.metrics.average_precision_at_k(
              labels=label_indices, predictions=logit, k=top_k)
          metrics.update({
              'metrics/midn_mAP_at_%i' % (top_k): (map_val, map_update)
          })
    return metrics
