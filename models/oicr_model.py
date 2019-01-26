from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from models.model_base import ModelBase
from protos import oicr_model_pb2

from nets import nets_factory
from core import imgproc
from core import utils
from core import plotlib
from core.standard_fields import InputDataFields
from core.standard_fields import OICRTasks
from core.standard_fields import OICRPredictions
from core.standard_fields import DetectionResultFields
from core.training_utils import build_hyperparams
from core import init_grid_anchors
from models import utils as model_utils
from core import box_utils

from object_detection.core.post_processing import batch_multiclass_non_max_suppression

slim = tf.contrib.slim


class Model(ModelBase):
  """OICR model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of oicr_model_pb2.OICRModel
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, oicr_model_pb2.OICRModel):
      raise ValueError('The model_proto has to be an instance of OICRModel.')

    self._vocabulary_list = model_utils.read_vocabulary(
        model_proto.vocabulary_file)
    self._num_classes = len(self._vocabulary_list)

  def _visl_proposals(self,
                      image,
                      num_proposals,
                      proposals,
                      top_k=100,
                      height=224,
                      width=224,
                      name='proposals'):
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
          image,
          boxes=proposals[:, :top_k, :],
          color=plotlib.RED,
          fontscale=1.0)
    tf.summary.image(name, image, max_outputs=5)

  def _visl_proposals_top_k(self,
                            image,
                            num_proposals,
                            proposals,
                            proposal_scores,
                            proposal_labels=None,
                            top_k=5,
                            threshold=0.01,
                            height=224,
                            width=224,
                            name='midn'):
    """Visualize top proposal results to the tensorboard.

    Args:
      image: A [batch, height, width, channels] float tensor, 
        ranging from 0 to 255.
      num_proposals: A [batch] int tensor.
      proposals: A [batch, max_num_proposals, 4] float tensor.
      proposal_scores: A [batch, max_num_proposals] float tensor.
      proposal_labels: A [batch, max_num_proposals] float tensor.
      height: Height of the visualized image.
      width: Width of the visualized image.
    """
    with tf.name_scope('visl_proposals'):
      image = tf.image.resize_images(image, [height, width])
      image = tf.cast(image, tf.uint8)

      (top_k_boxes, top_k_scores,
       top_k_labels) = model_utils.get_top_k_boxes_and_scores(
           proposals, proposal_scores, proposal_labels, k=top_k)

      proposal_scores = tf.where(proposal_scores > threshold, proposal_scores,
                                 -9999.0 * tf.ones_like(proposal_scores))
      image = plotlib.draw_rectangles(
          image,
          boxes=top_k_boxes,
          scores=top_k_scores,
          labels=top_k_labels,
          color=plotlib.RED,
          fontscale=1.0)
    tf.summary.image(name, image, max_outputs=5)

  def _calc_spp_feature(self, inputs, spp_bins=[1, 2, 3, 6], max_pool=True):
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

    pool_fn = tf.nn.avg_pool
    if max_pool:
      pool_fn = tf.nn.max_pool

    with tf.name_scope('calc_spp_feature'):
      pool_outputs = []
      for bins in spp_bins:
        if height % bins or width % bins:
          raise ValueError('Reminder should be ZERO.')

        pool_h, pool_w = height // bins, width // bins
        stride_h, stride_w = height // bins, width // bins
        pool = pool_fn(
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

  def _calc_conv_proposal_feature(self, image_feature_cropped):
    """Calculates proposal feature using spp.

    Args:
      image_feature_cropped: A [batch, crop_size, crop_size, feature_dims]
        float tensor.

    Returns:
      proposal_feature: A [batch, proposal_feature_dims] float tensor.
    """
    options = self._model_proto
    is_training = self._is_training

    with slim.arg_scope(
        build_hyperparams(options.conv_hyperparams, is_training)):

      net = image_feature_cropped

      with tf.variable_scope('conv_layers'):
        for i in range(options.conv_layers):
          with tf.variable_scope('layer_{}'.format(i)):
            tf.logging.info('layer input: %s', net.get_shape())
            net_cat = slim.conv2d(
                net,
                options.conv_units, [1, 1],
                stride=1,
                padding='SAME',
                scope='conv2d_1x1')
            net_cat = slim.dropout(
                net_cat,
                options.conv_dropout_keep_prob,
                is_training=is_training)

            net = tf.concat([net, net_cat], axis=-1)
            net = slim.max_pool2d(
                net, [2, 2], stride=2, padding='VALID', scope='maxpool_2x2')

            tf.logging.info('layer output: %s', net.get_shape())

      with tf.variable_scope('conv_layers'):
        with tf.variable_scope('layer_{}'.format(options.conv_layers)):
          net = slim.conv2d(
              net,
              options.conv_units, [3, 3],
              stride=1,
              padding='VALID',
              scope='conv2d_3x3')
          net = slim.dropout(
              net, options.conv_dropout_keep_prob, is_training=is_training)

      proposal_feature = tf.squeeze(net, [1, 2])

    tf.logging.info('proposal_feture: %s', proposal_feature)
    return proposal_feature

  def _calc_spp_proposal_feature(self, image_feature_cropped):
    """Calculates proposal feature using spp.

    Args:
      image_feature_cropped: A [batch, crop_size, crop_size, feature_dims]
        float tensor.

    Returns:
      spp_feature: A [batch, spp_feature_dims] float tensor.
      proposal_feature: A [batch, proposal_feature_dims] float tensor.
    """
    options = self._model_proto
    is_training = self._is_training

    net = image_feature_cropped

    #with slim.arg_scope(
    #    build_hyperparams(options.conv_hyperparams, is_training)):
    #  for i in range(options.conv_layers):
    #    net = slim.conv2d(
    #        net,
    #        options.conv_units, [1, 1],
    #        padding='SAME',
    #        scope='hidden/conv_{}'.format(i + 1))
    #    net = slim.dropout(
    #        net, options.conv_dropout_keep_prob, is_training=is_training)

    spp_feature = net = self._calc_spp_feature(
        net,
        spp_bins=[lv for lv in options.spp_bins],
        max_pool=options.spp_max_pool)

    for i in range(options.hidden_layers):
      with slim.arg_scope(
          build_hyperparams(options.fc_hyperparams, is_training)):
        net = slim.fully_connected(
            net,
            num_outputs=options.hidden_units,
            scope='hidden/fc_{}'.format(i + 1))
        net = slim.dropout(
            net, options.hidden_dropout_keep_prob, is_training=is_training)
    return spp_feature, net

  def _calc_vgg_proposal_feature(self, image_feature_cropped):
    """Calculates proposal feature using vgg fc layers.

    Args:
      image_feature_cropped: A [batch, crop_size, crop_size, feature_dims]
        float tensor.

    Returns:
      proposal_feature: A [batch, proposal_feature_dims] float tensor.
    """
    options = self._model_proto
    is_training = self._is_training

    # SPP.
    bins = 7
    batch, height, width, _ = utils.get_tensor_shape(image_feature_cropped)
    if height % bins or width % bins:
      raise ValueError('Reminder should be ZERO.')

    pool_h, pool_w = height // bins, width // bins
    stride_h, stride_w = height // bins, width // bins
    net = tf.nn.max_pool(
        image_feature_cropped,
        ksize=[1, pool_h, pool_w, 1],
        strides=[1, stride_h, stride_w, 1],
        padding='SAME')

    with tf.variable_scope(options.cnn.scope, reuse=True):
      with tf.variable_scope(options.cnn.name, reuse=True):

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
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')

    return net

  def _build_midn_network(self,
                          num_proposals,
                          spp_feature,
                          proposal_feature,
                          num_classes=20,
                          attention_normalizer=1.0,
                          attention_tanh=False,
                          attention_scale_factor=5.0):
    """Builds the Multiple Instance Detection Network.

    MIDN: An attention network.

    Args:
      num_proposals: A [batch] int tensor.
      spp_feature: A [batch, max_num_proposals, spp_feature_dims] 
        float tensor.
      proposal_feature: A [batch, max_num_proposals, proposal_feature_dims] 
        float tensor.
      num_classes: Number of classes.

    Returns:
      logits: A [batch, num_classes] float tensor.
      proba_r_given_c: A [batch, max_num_proposals, num_classes] float tensor.
    """
    with tf.name_scope('multi_instance_detection'):

      batch, max_num_proposals, _ = utils.get_tensor_shape(spp_feature)
      mask = tf.sequence_mask(
          num_proposals, maxlen=max_num_proposals, dtype=tf.float32)
      mask = tf.expand_dims(mask, axis=-1)

      spp_feature = tf.multiply(mask, spp_feature)
      proposal_feature = tf.multiply(mask, proposal_feature)

      # Calculates the score of proposal `r` given class `c`.
      #   proba_r_given_c shape = [batch, max_num_proposals, num_classes].

      logits_r_given_c = slim.fully_connected(
          proposal_feature,
          num_outputs=num_classes,
          activation_fn=None,
          scope='midn/proba_r_given_c')
      logits_r_given_c = tf.multiply(mask,
                                     logits_r_given_c / attention_normalizer)
      proba_r_given_c = utils.masked_softmax(
          data=logits_r_given_c, mask=mask, dim=1)

      # Calculates the score of class `c` given proposal `r`.
      #   proba_c_given_r shape = [batch, max_num_proposals, num_classes].

      logits_c_given_r = slim.fully_connected(
          spp_feature,
          num_outputs=num_classes,
          activation_fn=None,
          scope='midn/proba_c_given_r')
      logits_c_given_r = tf.multiply(mask,
                                     logits_c_given_r / attention_normalizer)

      # Aggregates the logits.

      logits = tf.multiply(logits_c_given_r, proba_r_given_c)
      logits = tf.reduce_sum(logits, axis=1)

    tf.summary.histogram('midn/logits_r_given_c', logits_r_given_c)
    tf.summary.histogram('midn/logits_c_given_r', logits_c_given_r)
    tf.summary.histogram('midn/logits', logits)

    return logits, proba_r_given_c

  def _post_process(self,
                    boxes,
                    scores,
                    score_thresh=1e-6,
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
                       prediction_task=OICRTasks.image_label,
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

    image_feature = model_utils.dilated_vgg16_conv(
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

    if options.feature_extractor == oicr_model_pb2.OICRModel.SPP:
      spp_feature, proposal_feature = self._calc_spp_proposal_feature(
          image_feature_cropped)
    elif options.feature_extractor == oicr_model_pb2.OICRModel.VGG:
      spp_feature = proposal_feature = self._calc_vgg_proposal_feature(
          image_feature_cropped)
    elif options.feature_extractor == oicr_model_pb2.OICRModel.CONV:
      spp_feature = proposal_feature = self._calc_conv_proposal_feature(
          image_feature_cropped)
    else:
      raise ValueError('Invalid feature extractor')

    spp_feature = tf.reshape(spp_feature, [batch, max_num_proposals, -1])
    proposal_feature = tf.reshape(proposal_feature,
                                  [batch, max_num_proposals, -1])

    tf.summary.histogram('midn/proposal_feature', proposal_feature)

    # Build the MIDN network.
    #   proba_r_given_c shape = [batch, max_num_proposals, num_classes].
    #   See `Multiple Instance Detection Network with OICR`.

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      midn_logits, proba_r_given_c = self._build_midn_network(
          num_proposals,
          spp_feature if options.use_spp_to_calc_logits else proposal_feature,
          proposal_feature,
          num_classes=self._num_classes,
          attention_normalizer=options.attention_normalizer,
          attention_tanh=options.attention_tanh,
          attention_scale_factor=options.attention_scale_factor)

    # Build the OICR network.
    #   proposal_scores shape = [batch, max_num_proposals, 1 + num_classes].
    #   See `Multiple Instance Detection Network with OICR`.

    oicr_proposal_scores_list = []
    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      with tf.name_scope('online_instance_classifier_refinement'):
        for i in range(options.oicr_iterations):
          oicr_proposal_scores_at_i = slim.fully_connected(
              proposal_feature,
              num_outputs=1 + self._num_classes,
              activation_fn=None,
              scope='oicr/iter{}'.format(i + 1))
          oicr_proposal_scores_list.append(oicr_proposal_scores_at_i)

    predictions = {
        DetectionResultFields.num_proposals: num_proposals,
        DetectionResultFields.proposal_boxes: proposals,
        OICRPredictions.midn_proba_r_given_c: proba_r_given_c,
        OICRPredictions.midn_logits: midn_logits,
    }

    # Post process to get the final detections.
    labels = self._extract_class_label(
        class_texts=examples[InputDataFields.caption_strings],
        vocabulary_list=self._vocabulary_list)

    midn_proposal_scores = tf.multiply(proba_r_given_c,
                                       tf.expand_dims(labels, axis=1))

    (predictions[DetectionResultFields.num_detections + '_at_{}'.format(0)],
     predictions[DetectionResultFields.detection_boxes + '_at_{}'.format(0)],
     predictions[DetectionResultFields.detection_scores + '_at_{}'.format(0)],
     predictions[DetectionResultFields.detection_classes +
                 '_at_{}'.format(0)]) = self._post_process(
                     proposals, midn_proposal_scores)

    for i, oicr_proposal_scores_at_i in enumerate(oicr_proposal_scores_list):
      predictions[OICRPredictions.oicr_proposal_scores +
                  '_at_{}'.format(i + 1)] = oicr_proposal_scores_at_i

      (predictions[DetectionResultFields.num_detections +
                   '_at_{}'.format(i + 1)],
       predictions[DetectionResultFields.detection_boxes +
                   '_at_{}'.format(i + 1)],
       predictions[DetectionResultFields.detection_scores +
                   '_at_{}'.format(i + 1)],
       predictions[DetectionResultFields.detection_classes +
                   '_at_{}'.format(i + 1)]) = self._post_process(
                       proposals,
                       tf.nn.softmax(oicr_proposal_scores_at_i,
                                     axis=-1)[:, :, 1:])

    self._visl_proposals(
        image, num_proposals, proposals, name='proposals', top_k=2000)
    for i in range(1 + options.oicr_iterations):
      num_detections, detection_boxes, detection_scores, detection_classes = (
          predictions[DetectionResultFields.num_detections +
                      '_at_{}'.format(i)],
          predictions[DetectionResultFields.detection_boxes +
                      '_at_{}'.format(i)],
          predictions[DetectionResultFields.detection_scores +
                      '_at_{}'.format(i)],
          predictions[DetectionResultFields.detection_classes +
                      '_at_{}'.format(i)])
      self._visl_proposals_top_k(
          image,
          num_detections,
          detection_boxes,
          detection_scores,
          tf.gather(self._vocabulary_list, tf.to_int32(detection_classes - 1)),
          name='detection_{}'.format(i))

    return predictions

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

  def _calc_oicr_loss(self,
                      labels,
                      num_proposals,
                      proposals,
                      scores_0,
                      scores_1,
                      scope,
                      iou_threshold=0.5):
    """Calculates the OICR loss at refinement stage `i`.

    Args:
      labels: A [batch, num_classes] float tensor.
      num_proposals: A [batch] int tensor.
      proposals: A [batch, max_num_proposals, 4] float tensor.
      scores_0: A [batch, max_num_proposal, 1 + num_classes] float tensor, 
        representing the proposal score at `k-th` refinement.
      scores_1: A [batch, max_num_proposal, 1 + num_classes] float tensor,
        representing the proposal score at `(k+1)-th` refinement.

    Returns:
      oicr_cross_entropy_loss: a scalar float tensor.
    """
    with tf.name_scope(scope):
      (batch, max_num_proposals,
       num_classes_plus_one) = utils.get_tensor_shape(scores_0)
      num_classes = num_classes_plus_one - 1

      # For each class, look for the most confident proposal.
      #   proposal_ind shape = [batch, num_classes].

      proposal_mask = tf.sequence_mask(
          num_proposals, maxlen=max_num_proposals, dtype=tf.float32)
      proposal_ind = utils.masked_argmax(
          tf.nn.softmax(scores_0, axis=-1)[:, :, 1:],
          tf.expand_dims(proposal_mask, axis=-1),
          dim=1)

      # Deal with the most confident proposal per each class.
      #   Unstack the `proposal_ind`, `labels`.
      #   proposal_labels shape = [batch, max_num_proposals, num_classes].

      proposal_labels = []
      indices_0 = tf.range(batch, dtype=tf.int64)
      for indices_1, label_per_class in zip(
          tf.unstack(proposal_ind, axis=-1), tf.unstack(labels, axis=-1)):

        # Gather the most confident proposal for the class.
        #   confident_proosal shape = [batch, 4].

        indices = tf.stack([indices_0, indices_1], axis=-1)
        confident_proposal = tf.gather_nd(proposals, indices)

        # Get the Iou from all the proposals to the most confident proposal.
        #   iou shape = [batch, max_num_proposals].

        confident_proposal_tiled = tf.tile(
            tf.expand_dims(confident_proposal, axis=1),
            [1, max_num_proposals, 1])
        iou = box_utils.iou(
            tf.reshape(proposals, [-1, 4]),
            tf.reshape(confident_proposal_tiled, [-1, 4]))
        iou = tf.reshape(iou, [batch, max_num_proposals])

        # Filter out irrelevant predictions using image-level label.

        target = tf.to_float(tf.greater_equal(iou, iou_threshold))
        target = tf.where(
            label_per_class > 0, x=target, y=tf.zeros_like(target))
        proposal_labels.append(target)

      proposal_labels = tf.stack(proposal_labels, axis=-1)

      # Add background targets, and normalize the sum value to 1.0.
      #   proposal_labels shape = [batch, max_num_proposals, 1 + num_classes].

      bkg = tf.logical_not(tf.reduce_sum(proposal_labels, axis=-1) > 0)
      proposal_labels = tf.concat(
          [tf.expand_dims(tf.to_float(bkg), axis=-1), proposal_labels], axis=-1)

      proposal_labels = tf.div(
          proposal_labels, tf.reduce_sum(
              proposal_labels, axis=-1, keepdims=True))

      assert_op = tf.Assert(
          tf.reduce_all(
              tf.abs(tf.reduce_sum(proposal_labels, axis=-1) - 1) < 1e-6),
          ["Probabilities not sum to ONE", proposal_labels])

      # Compute the loss.

      with tf.control_dependencies([assert_op]):
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=proposal_labels, logits=scores_1)
        oicr_cross_entropy_loss = tf.reduce_mean(
            utils.masked_avg(data=losses, mask=proposal_mask, dim=1))

    return oicr_cross_entropy_loss

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

      # Extract image-level labels.

      labels = self._extract_class_label(
          class_texts=examples[InputDataFields.caption_strings],
          vocabulary_list=self._vocabulary_list)

      # Loss of the multi-instance detection network.

      midn_logits = predictions[OICRPredictions.midn_logits]
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=midn_logits)
      loss_dict['midn_cross_entropy_loss'] = tf.reduce_mean(losses)

      # Losses of the online instance classifier refinement network.

      options = self._model_proto

      (num_proposals, proposals,
       proposal_scores_0) = (predictions[DetectionResultFields.num_proposals],
                             predictions[DetectionResultFields.proposal_boxes],
                             predictions[OICRPredictions.midn_proba_r_given_c])

      batch, max_num_proposals, _ = utils.get_tensor_shape(proposal_scores_0)
      proposal_scores_0 = tf.concat(
          [tf.fill([batch, max_num_proposals, 1], 0.0), proposal_scores_0],
          axis=-1)

      for i in range(options.oicr_iterations):
        proposal_scores_1 = predictions[OICRPredictions.oicr_proposal_scores +
                                        '_at_{}'.format(i + 1)]
        loss_dict['oicr_cross_entropy_loss_at_{}'.format(
            i + 1)] = self._calc_oicr_loss(
                labels,
                num_proposals,
                proposals,
                proposal_scores_0,
                proposal_scores_1,
                scope='oicr_{}'.format(i + 1),
                iou_threshold=options.oicr_iou_threshold)

        proposal_scores_0 = proposal_scores_1

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
