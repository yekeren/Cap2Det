from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from core import utils
from core import box_utils
from core.training_utils import build_hyperparams
from object_detection.builders.model_builder import _build_faster_rcnn_feature_extractor as build_faster_rcnn_feature_extractor

slim = tf.contrib.slim


def calc_oicr_loss(labels,
                   num_proposals,
                   proposals,
                   scores_0,
                   scores_1,
                   scope,
                   iou_threshold=0.5):
  """Calculates the NOD loss at refinement stage `i`.

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

    proposal_mask = tf.sequence_mask(num_proposals,
                                     maxlen=max_num_proposals,
                                     dtype=tf.float32)
    proposal_ind = utils.masked_argmax(scores_0[:, :, 1:],
                                       tf.expand_dims(proposal_mask, axis=-1),
                                       dim=1)

    # Deal with the most confident proposal per each class.
    #   Unstack the `proposal_ind`, `labels`.
    #   proposal_labels shape = [batch, max_num_proposals, num_classes].

    proposal_labels = []
    indices_0 = tf.range(batch, dtype=tf.int64)
    for indices_1, label_per_class in zip(tf.unstack(proposal_ind, axis=-1),
                                          tf.unstack(labels, axis=-1)):

      # Gather the most confident proposal for the class.
      #   confident_proosal shape = [batch, 4].

      indices = tf.stack([indices_0, indices_1], axis=-1)
      confident_proposal = tf.gather_nd(proposals, indices)

      # Get the Iou from all the proposals to the most confident proposal.
      #   iou shape = [batch, max_num_proposals].

      confident_proposal_tiled = tf.tile(
          tf.expand_dims(confident_proposal, axis=1), [1, max_num_proposals, 1])
      iou = box_utils.iou(tf.reshape(proposals, [-1, 4]),
                          tf.reshape(confident_proposal_tiled, [-1, 4]))
      iou = tf.reshape(iou, [batch, max_num_proposals])

      # Filter out irrelevant predictions using image-level label.

      target = tf.to_float(tf.greater_equal(iou, iou_threshold))
      target = tf.where(label_per_class > 0, x=target, y=tf.zeros_like(target))
      proposal_labels.append(target)

    proposal_labels = tf.stack(proposal_labels, axis=-1)

    # Add background targets, and normalize the sum value to 1.0.
    #   proposal_labels shape = [batch, max_num_proposals, 1 + num_classes].

    bkg = tf.logical_not(tf.reduce_sum(proposal_labels, axis=-1) > 0)
    proposal_labels = tf.concat(
        [tf.expand_dims(tf.to_float(bkg), axis=-1), proposal_labels], axis=-1)

    proposal_labels = tf.div(
        proposal_labels, tf.reduce_sum(proposal_labels, axis=-1, keepdims=True))

    assert_op = tf.Assert(
        tf.reduce_all(
            tf.abs(tf.reduce_sum(proposal_labels, axis=-1) - 1) < 1e-6),
        ["Probabilities not sum to ONE", proposal_labels])

    # Compute the loss.

    with tf.control_dependencies([assert_op]):
      losses = tf.nn.softmax_cross_entropy_with_logits(
          labels=tf.stop_gradient(proposal_labels), logits=scores_1)
      # oicr_cross_entropy_loss = tf.reduce_mean(
      #     utils.masked_avg(data=losses, mask=proposal_mask, dim=1))
      oicr_cross_entropy_losses = utils.masked_avg(data=losses,
                                                   mask=proposal_mask,
                                                   dim=1)

  return tf.squeeze(oicr_cross_entropy_losses, 1)


def extract_frcnn_feature(inputs,
                          num_proposals,
                          proposals,
                          options,
                          is_training=False):
  """Extracts Fast-RCNN feature from image.

  Args:
    feature_extractor: An FRCNN feature extractor instance.
    inputs: A [batch, height, width, channels] float tensor.
    num_proposals: A [batch] int tensor.
    proposals: A [batch, max_num_proposals, 4] float tensor.
    options:
    is_training:

  Returns:
    proposal_features: A [batch, max_num_proposals, feature_dims] float 
      tensor.
  """
  feature_extractor = build_faster_rcnn_feature_extractor(
      options.feature_extractor, is_training, options.inplace_batchnorm_update)

  # Extract `features_to_crop` from the original image.
  #   shape = [batch, feature_height, feature_width, feature_depth].

  preprocessed_inputs = feature_extractor.preprocess(inputs)

  (features_to_crop, _) = feature_extractor.extract_proposal_features(
      preprocessed_inputs, scope='first_stage_feature_extraction')

  if options.dropout_on_feature_map:
    features_to_crop = slim.dropout(features_to_crop,
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

  (box_classifier_features) = feature_extractor.extract_box_classifier_features(
      flattened_proposal_features_maps, scope='second_stage_feature_extraction')

  flattened_roi_pooled_features = tf.reduce_mean(box_classifier_features,
                                                 [1, 2],
                                                 name='AvgPool')
  flattened_roi_pooled_features = slim.dropout(
      flattened_roi_pooled_features,
      keep_prob=options.dropout_keep_prob,
      is_training=is_training)

  proposal_features = tf.reshape(flattened_roi_pooled_features,
                                 [batch, max_num_proposals, -1])

  # Allow to train from scratch (resolving journal review comments).
  if not options.checkpoint_path:
    return proposal_features

  # Assign weights from pre-trained checkpoint.

  if not options.from_detection_checkpoint:
    tf.train.init_from_checkpoint(
        options.checkpoint_path,
        assignment_map={"/": "first_stage_feature_extraction/"})
    tf.train.init_from_checkpoint(
        options.checkpoint_path,
        assignment_map={"/": "second_stage_feature_extraction/"})
  else:
    tf.train.init_from_checkpoint(options.checkpoint_path,
                                  assignment_map={
                                      "FirstStageFeatureExtractor/":
                                          "first_stage_feature_extraction/"
                                  })
    tf.train.init_from_checkpoint(options.checkpoint_path,
                                  assignment_map={
                                      "SecondStageFeatureExtractor/":
                                          "second_stage_feature_extraction/"
                                  })
    if options.HasField('projection_layer'):
      projection_layer_config = options.projection_layer

      with slim.arg_scope(
          build_hyperparams(projection_layer_config.fc_hyperparams,
                            is_training=is_training)):
        proposal_features = slim.fully_connected(
            proposal_features,
            projection_layer_config.output_dims,
            activation_fn=None,
            scope=projection_layer_config.scope)

      tf.train.init_from_checkpoint(options.checkpoint_path,
                                    assignment_map={
                                        projection_layer_config.scope + '/':
                                            projection_layer_config.scope + '/'
                                    })

  return proposal_features
