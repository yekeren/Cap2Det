from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from models.model_base import ModelBase
from protos import wsod_voc_model_pb2

from nets import nets_factory
from nets import vgg
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

from object_detection.builders import hyperparams_builder
from object_detection.builders import box_predictor_builder
from object_detection.core.post_processing import batch_multiclass_non_max_suppression
from object_detection.builders.model_builder import _build_faster_rcnn_feature_extractor as build_faster_rcnn_feature_extractor

slim = tf.contrib.slim
_BIG_NUMBER = 1e8


class Model(ModelBase):
  """OICR model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of wsod_voc_model_pb2.WsodVocModel
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, wsod_voc_model_pb2.WsodVocModel):
      raise ValueError('The model_proto has to be an instance of WsodVocModel.')

    options = model_proto

    self._vocabulary_list = model_utils.read_vocabulary(options.vocabulary_file)

    self._num_classes = len(self._vocabulary_list)

    self._feature_extractor = build_faster_rcnn_feature_extractor(
        options.feature_extractor, is_training,
        options.inplace_batchnorm_update)

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

  def _visl_class_activation_map(self,
                                 image,
                                 class_activation_map,
                                 height=224,
                                 width=224):
    """Visualize class activation map to the tensorboard.

    Args:
      image: A [batch, height, width, channels] float tensor, 
        ranging from 0 to 255.
      class_activation_map: A [batch, height, width, num_classes] float tensor.
      height: Height of the visualized image.
      width: Width of the visualized image.
    """
    options = self._model_proto

    batch, _, _, _ = utils.get_tensor_shape(image)

    # Initialize text labels to attach to images.

    vocabulary_list = [
        tf.tile(tf.expand_dims(w, axis=0), [batch])
        for w in self._vocabulary_list
    ]

    merge_v_fn = lambda x: tf.concat(x, axis=1)
    merge_h_fn = lambda x: tf.concat(x, axis=2)

    # Visualize heat map from different resolutions.

    image_visl = tf.cast(
        tf.image.resize_images(image, [height * 2, width * 2]), tf.uint8)

    min_v = tf.reduce_min(class_activation_map, axis=[1, 2, 3], keepdims=True)
    max_v = tf.reduce_max(class_activation_map, axis=[1, 2, 3], keepdims=True)
    class_activation_map = (class_activation_map - min_v) / (max_v - min_v)

    visl_list_at_i = []
    for class_id, x in enumerate(
        tf.unstack(
            tf.image.resize_images(class_activation_map, [height, width]),
            axis=-1)):
      class_name = vocabulary_list[class_id]

      # Draw class-related heat map.

      x = plotlib.convert_to_heatmap(x, normalize=False)
      x = tf.image.convert_image_dtype(x, tf.uint8)
      x = plotlib.draw_caption(x, class_name, org=(0, 0), color=plotlib.RED)
      visl_list_at_i.append(x)

    half_size = len(visl_list_at_i) // 2
    visl_image = merge_h_fn([image_visl] + [
        merge_v_fn([
            merge_h_fn(visl_list_at_i[:half_size]),
            merge_h_fn(visl_list_at_i[half_size:])
        ])
    ])

    tf.summary.image("heatmap", visl_image, max_outputs=5)

  def _calc_anchor_scores(self,
                          class_activation_map,
                          anchors,
                          resize_height=224,
                          resize_width=224,
                          num_boxes_per_class=100):
    """Calculates class activation box based on the class activation map.

    Args:
      class_act_map: A [batch, height, width, num_classes] float tensor.
      anchor_boxes: A [batch, number_of_anchors, 4] float tensor.

    Returns:
      anchor_scores: A [batch, number_of_anchors, num_classes] tensor.
    """
    with tf.name_scope('calc_anchor_scores'):
      class_activation_map = tf.image.resize_images(
          class_activation_map, [resize_height, resize_width])
      batch, height, width, num_classes = utils.get_tensor_shape(
          class_activation_map)
      ymin, xmin, ymax, xmax = tf.unstack(anchors, axis=-1)
      anchors_absolute = tf.stack([
          tf.to_int64(tf.round(ymin * tf.to_float(height))),
          tf.to_int64(tf.round(xmin * tf.to_float(width))),
          tf.to_int64(tf.round(ymax * tf.to_float(height))),
          tf.to_int64(tf.round(xmax * tf.to_float(width)))
      ],
                                  axis=-1)

      fn = model_utils.build_proposal_saliency_fn(
          func_name='wei', border_ratio=0.2, purity_weight=1.0)
      anchor_scores = fn(class_activation_map, anchors_absolute)
    return anchor_scores

  def _build_mipn_network(self,
                          features_to_crop,
                          num_proposals,
                          proposals,
                          kernel_size=1,
                          pooling=None,
                          num_classes=20):
    """Builds the Multiple Instance Proposal Network.

    Args:
      features_to_crop: A [batch, feature_height, feature_width, feature_dims] 
        float tensor.
      num_proposals: A [batch] int tensor.
      proposals: A [batch, max_num_proposals, 4] float tensor.
      num_classes: Number of classes.

    Returns:
      mipn_logits: a [batch, num_classes] float tensor.
      mipn_num_proposals: A [batch] int tensor.
      mipn_proposals: A [batch, max_num_proposals, 4] float tensor.
    """
    batch, feature_height, feature_width, _ = utils.get_tensor_shape(
        features_to_crop)
    _, max_num_proposals, _ = utils.get_tensor_shape(proposals)

    # Calculates the logits per pixel:
    #   shape = [batch, feature_height, feature_width, num_classes].

    logits_c_given_p = tf.contrib.layers.conv2d(
        inputs=features_to_crop,
        num_outputs=num_classes,
        kernel_size=[kernel_size, kernel_size],
        activation_fn=None,
        scope='mipn/proba_c_given_p')
    class_activation_scores = tf.nn.softmax(logits_c_given_p)

    # Computes the `proposal_score`, shape = [batch, max_num_proposals].

    proposal_scores = self._calc_anchor_scores(class_activation_scores,
                                               proposals)
    proposal_scores = tf.reduce_max(proposal_scores, axis=-1)

    # Sort the proposals.
    mask = tf.sequence_mask(
        num_proposals, maxlen=max_num_proposals, dtype=tf.float32)
    proposal_scores = proposal_scores - _BIG_NUMBER * (1.0 - mask)

    indices_0 = tf.expand_dims(tf.range(batch, dtype=tf.int32), axis=1)
    indices_0 = tf.tile(indices_0, [1, max_num_proposals])
    indices_1 = tf.contrib.framework.argsort(
        proposal_scores, axis=-1, direction='DESCENDING')
    indices = tf.stack([indices_0, indices_1], axis=-1)

    proposals = tf.gather_nd(proposals, indices)
    proposal_scores = tf.gather_nd(proposal_scores, indices)

    if pooling == wsod_voc_model_pb2.WsodVocModel.AVG:
      logits = tf.reduce_mean(logits_c_given_p, axis=[1, 2])
    elif pooling == wsod_voc_model_pb2.WsodVocModel.MAX:
      logits = tf.reduce_max(logits_c_given_p, axis=[1, 2])

    return (logits, num_proposals, proposals, proposal_scores,
            class_activation_scores)

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

      # Calculates the attention score: proposal `r` given class `c`.
      #   proba_r_given_c shape = [batch, max_num_proposals, num_classes].

      logits_r_given_c = slim.fully_connected(
          proposal_features,
          num_outputs=num_classes,
          activation_fn=None,
          scope='midn/proba_r_given_c')
      logits_r_given_c = tf.multiply(mask, logits_r_given_c)
      proba_r_given_c = utils.masked_softmax(
          data=logits_r_given_c, mask=mask, dim=1)
      proba_r_given_c = tf.multiply(mask, proba_r_given_c)
      tf.summary.histogram('midn/logits_r_given_c', logits_r_given_c)

      # Calculates the weighted logits:
      #   logits_c_given_r shape = [batch, max_num_proposals, num_classes].
      #   logits shape = [batch, num_classes].

      logits_c_given_r = slim.fully_connected(
          proposal_features,
          num_outputs=num_classes,
          activation_fn=None,
          scope='midn/proba_c_given_r')
      # proba_c_given_r = tf.nn.softmax(logits_c_given_r)
      # proba_c_given_r = tf.multiply(mask, proba_c_given_r)
      tf.summary.histogram('midn/logits_c_given_r', logits_c_given_r)

      # Aggregates the logits.

      logits = tf.multiply(logits_c_given_r, proba_r_given_c)
      logits = tf.reduce_sum(logits, axis=1)
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

    (inputs, num_proposals,
     proposals) = (examples[InputDataFields.image],
                   examples[InputDataFields.num_proposals],
                   examples[InputDataFields.proposals])

    # Extract `features_to_crop` from the original image.
    #   shape = [batch, feature_height, feature_width, feature_depth].

    preprocessed_inputs = self._feature_extractor.preprocess(inputs)

    (features_to_crop, _) = self._feature_extractor.extract_proposal_features(
        preprocessed_inputs, scope='first_stage_feature_extraction')
    (mipn_feature_map
    ) = self._feature_extractor.extract_box_classifier_features(
        features_to_crop, scope='second_stage_feature_extraction')

    with slim.arg_scope(
        build_hyperparams(options.conv_hyperparams, is_training)):
      (mipn_logits, mipn_num_proposals, mipn_proposals, mipn_proposal_scores,
       class_activation_map) = self._build_mipn_network(
           mipn_feature_map,
           num_proposals,
           proposals,
           kernel_size=options.mipn_conv_kernel_size,
           pooling=options.mipn_pooling)

    self._visl_class_activation_map(inputs, class_activation_map)

    self._visl_proposals(
        inputs, num_proposals, proposals, name='proposals', top_k=200)
    self._visl_proposals(
        inputs,
        mipn_num_proposals,
        mipn_proposals,
        name='proposals_mipn',
        top_k=200)

    # Substitude to use the top-ranked proposals.

    num_proposals = tf.minimum(mipn_num_proposals,
                               options.mipn_max_num_proposals)
    proposals = mipn_proposals[:, :options.mipn_max_num_proposals, :]

    batch, max_num_proposals, _ = utils.get_tensor_shape(proposals)

    # Crop `flattened_proposal_features_maps`.
    #   shape = [batch*max_num_proposals, crop_size, crop_size, feature_depth].

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

    # Build MIDN network.
    #   proba_r_given_c shape = [batch, max_num_proposals, num_classes].

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      midn_logits, proba_r_given_c = self._build_midn_network(
          num_proposals, proposal_features, num_classes=self._num_classes)

    # Build the OICR network.
    #   proposal_scores shape = [batch, max_num_proposals, 1 + num_classes].
    #   See `Multiple Instance Detection Network with OICR`.

    oicr_proposal_scores_list = []
    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      with tf.name_scope('online_instance_classifier_refinement'):
        for i in range(options.oicr_iterations):
          oicr_proposal_scores_at_i = slim.fully_connected(
              proposal_features,
              num_outputs=1 + self._num_classes,
              activation_fn=None,
              scope='oicr/iter{}'.format(i + 1))
          oicr_proposal_scores_list.append(oicr_proposal_scores_at_i)

    predictions = {
        DetectionResultFields.num_proposals: num_proposals,
        DetectionResultFields.proposal_boxes: proposals,
        OICRPredictions.midn_logits: midn_logits,
        OICRPredictions.midn_proba_r_given_c: proba_r_given_c,
        OICRPredictions.mipn_logits: mipn_logits,
    }

    # Post process to get the final detections.

    midn_proposal_scores = tf.multiply(
        tf.expand_dims(tf.nn.softmax(midn_logits), axis=1), proba_r_given_c)

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
          inputs,
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
            labels=tf.stop_gradient(proposal_labels), logits=scores_1)
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
    options = self._model_proto
    loss_dict = {}

    with tf.name_scope('losses'):

      # Extract image-level labels.

      labels = self._extract_class_label(
          class_texts=examples[InputDataFields.caption_strings],
          vocabulary_list=self._vocabulary_list)

      # Loss of the multi-instance proposal network.

      mipn_logits = predictions[OICRPredictions.mipn_logits]
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=mipn_logits)
      loss_dict['mipn_cross_entropy_loss'] = tf.multiply(
          options.mipn_loss_weight, tf.reduce_mean(losses))

      # Loss of the multi-instance detection network.

      midn_logits = predictions[OICRPredictions.midn_logits]
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=midn_logits)
      loss_dict['midn_cross_entropy_loss'] = tf.multiply(
          options.midn_loss_weight, tf.reduce_mean(losses))

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

      global_step = tf.train.get_or_create_global_step()
      oicr_loss_mask = tf.cast(global_step > options.oicr_start_step,
                               tf.float32)

      for i in range(options.oicr_iterations):
        proposal_scores_1 = predictions[OICRPredictions.oicr_proposal_scores +
                                        '_at_{}'.format(i + 1)]
        oicr_cross_entropy_loss_at_i = self._calc_oicr_loss(
            labels,
            num_proposals,
            proposals,
            proposal_scores_0,
            proposal_scores_1,
            scope='oicr_{}'.format(i + 1),
            iou_threshold=options.oicr_iou_threshold)
        loss_dict['oicr_cross_entropy_loss_at_{}'.format(
            i + 1)] = oicr_loss_mask * oicr_cross_entropy_loss_at_i

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
