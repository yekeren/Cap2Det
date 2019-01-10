from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from models.model_base import ModelBase
from protos import cam_model_pb2

from core import imgproc
from core import utils
from core import plotlib
from core.standard_fields import InputDataFields
from core.standard_fields import CAMTasks
from core.standard_fields import CAMPredictions
from core.standard_fields import CAMVariableScopes
from core.training_utils import build_hyperparams
from core import init_grid_anchors
from models import utils as model_utils

slim = tf.contrib.slim


class Model(ModelBase):
  """GAP model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of cam_model_pb2.CAMModel
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, cam_model_pb2.CAMModel):
      raise ValueError('The model_proto has to be an instance of CAMModel.')

    self._vocabulary_list = model_utils.read_vocabulary(
        model_proto.vocabulary_file)
    tf.logging.info('Load %i classes: %s', len(self._vocabulary_list),
                    ','.join(self._vocabulary_list))

    self._input_scales = [1.0]
    if len(model_proto.input_image_scale) > 0:
      self._input_scales = [scale for scale in model_proto.input_image_scale]

    self._cnn_feature_names = [model_proto.cnn_output_name]
    if len(model_proto.cnn_feature_name) > 0:
      self._cnn_feature_names = [name for name in model_proto.cnn_feature_name]

    self._anchors = init_grid_anchors.initialize_grid_anchors(stride_ratio=0.2)

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      a list of model variables or None by default.
    """
    options = self._model_proto

    # Filter out CNN variables that are not trainable.
    variables_to_train = []
    for var in tf.trainable_variables():
      if not options.cnn_trainable:
        if CAMVariableScopes.cnn in var.op.name:
          tf.logging.info("Freeze cnn parameter %s.", var.op.name)
          continue
      variables_to_train.append(var)

    for var in variables_to_train:
      tf.logging.info("Model variables: %s.", var.op.name)
    return variables_to_train

  def _project_image(self,
                     image_feature,
                     num_outputs=20,
                     kernel_size=1,
                     hyperparams=None,
                     is_training=False):
    """Adds additional 1x1 conv layer to project image features.

    Args:
      image_feature: [batch, feature_height, feature_width, feature_depth] float
        tensor, which is the CNN output.
      num_outputs: number of output channels.
      hyperparams: an instance of hyperparams_pb2.Hyperparams, used for the
        conv2d projection layer.
      is_training: if True, training graph is built.
    """
    with slim.arg_scope(build_hyperparams(hyperparams, is_training)):
      output = tf.contrib.layers.conv2d(
          inputs=image_feature,
          num_outputs=num_outputs,
          kernel_size=[kernel_size, kernel_size],
          activation_fn=None)
    return output

  def _calc_frcnn_feature(self, preprocessed_image, num_classes):
    """Computes multi-resolutional feature map using the method in FRCNN.

    Args:
      preprocessed_image: a [batch, height, width, channels] float tensor.
      num_classes: number of classes.

    Returns:
      cnn_feature_map_list: a list of cnn feature maps, each of them is a
        [batch, feature_height, feature_width, feature_dims] float tensor.
      class_act_map_list: a list of class activation maps, each of them is a
        [batch, feature_height, feature_width, num_classes] float tensor.
    """
    options = self._model_proto
    is_training = self._is_training

    batch, height, width, _ = utils.get_tensor_shape(preprocessed_image)

    reuse_cnn = False
    cnn_feature_map_list, class_act_map_list = [], []

    for scale in self._input_scales:

      # Resize the input images.

      if scale == 1.0:
        resized_image = preprocessed_image
      else:
        new_height = tf.to_int32(tf.round(tf.to_float(height) * scale))
        new_width = tf.to_int32(tf.round(tf.to_float(width) * scale))
        resized_image = tf.image.resize_images(
            preprocessed_image, tf.stack([new_height, new_width]))

      # Extract CNN feature map.

      with tf.variable_scope(CAMVariableScopes.cnn, reuse=reuse_cnn):
        image_feature = model_utils.extract_image_feature(
            resized_image,
            name=options.cnn_name,
            weight_decay=options.cnn_weight_decay,
            output_name=options.cnn_output_name,
            is_training=is_training and options.cnn_trainable)
        image_feature = slim.dropout(
            image_feature,
            options.cnn_dropout_keep_prob,
            is_training=is_training)

      # Add an extra 1x1 conv layer to project image feature map.

      with tf.variable_scope(CAMVariableScopes.image_proj, reuse=reuse_cnn):
        class_act_map = self._project_image(
            image_feature,
            num_outputs=num_classes,
            kernel_size=options.image_proj_kernel_size,
            hyperparams=options.image_proj_hyperparams,
            is_training=is_training)

      cnn_feature_map_list.append(image_feature)
      class_act_map_list.append(class_act_map)
      reuse_cnn = True

    return cnn_feature_map_list, class_act_map_list

  def _calc_ssd_feature(self, preprocessed_image, num_classes):
    """Computes multi-resolutional feature map using the method in SSD.

    Args:
      preprocessed_image: a [batch, height, width, channels] float tensor.
      num_classes: number of classes.

    Returns:
      cnn_feature_map_list: a list of cnn feature maps, each of them is a
        [batch, feature_height, feature_width, feature_dims] float tensor.
      class_act_map_list: a list of class activation maps, each of them is a
        [batch, feature_height, feature_width, num_classes] float tensor.
    """
    options = self._model_proto
    is_training = self._is_training

    with tf.variable_scope(CAMVariableScopes.cnn):
      cnn_feature_map_list = model_utils.extract_image_feature(
          preprocessed_image,
          name=options.cnn_name,
          weight_decay=options.cnn_weight_decay,
          output_name=self._cnn_feature_names,
          is_training=is_training and options.cnn_trainable)
      cnn_feature_map_list = [
          slim.dropout(
              image_feature,
              options.cnn_dropout_keep_prob,
              is_training=is_training) for image_feature in cnn_feature_map_list
      ]

    class_act_map_list = []
    for image_feature, layer_name in zip(cnn_feature_map_list,
                                         self._cnn_feature_names):
      layer_name = layer_name.split('/')[-1] + '_proj'
      with tf.variable_scope(layer_name):
        class_act_map = self._project_image(
            image_feature,
            num_outputs=num_classes,
            hyperparams=options.image_proj_hyperparams,
            is_training=is_training)
      class_act_map_list.append(class_act_map)

    return cnn_feature_map_list, class_act_map_list

  def _calc_class_activation_map_list(self, image, num_classes):
    """Builds tf graph to predict class activation map.

    Args:
      image: a [batch, height, width, channels] float tensor, ranging 
        from 0 to 255.
      num_classes: number of classes.

    Returns:
      cnn_feature_map_list: a list of cnn feature maps, each of them is a
        [batch, feature_height, feature_width, feature_dims] float tensor.
      class_act_map_list: a list of class activation maps, each of them is a
        [batch, feature_height, feature_width, num_classes] float tensor.
    """
    options = self._model_proto
    is_training = self._is_training

    preprocessed_image = model_utils.preprocess_image(image,
                                                      options.preprocess_method)

    # Get the multi-scale CNN feature map.

    if options.feature_extractor == cam_model_pb2.CAMModel.FRCNN:
      cnn_feature_map_list, class_act_map_list = self._calc_frcnn_feature(
          preprocessed_image, num_classes)
    elif options.feature_extractor == cam_model_pb2.CAMModel.SSD:
      cnn_feature_map_list, class_act_map_list = self._calc_ssd_feature(
          preprocessed_image, num_classes)
    else:
      raise ValueError('Invalid feature extractor {}'.format(
          options.feature_extractor))

    if options.cnn_checkpoint:
      tf.train.init_from_checkpoint(
          options.cnn_checkpoint,
          assignment_map={"/": CAMVariableScopes.cnn + "/"})

    return cnn_feature_map_list, class_act_map_list

  def _visl_proposals(self,
                      image,
                      number_of_proposals,
                      proposals,
                      proposal_scores,
                      vocabulary_list,
                      height=448,
                      width=448):
    """Visualize proposal results to the tensorboard.

    Args:
      image: A [batch, height, width, channels] float tensor, 
        ranging from 0 to 255.
      number_of_proposals: A [batch] int tensor.
      proposals: A [batch, max_num_proposals, 4] float tensor.
      proposal_scores: A [batch, max_num_proposals, num_classes] float tensor.
      vocabulary_list: Names of the classes.
      height: Height of the visualized image.
      width: Width of the visualized image.
    """
    proposal_labels = tf.gather(vocabulary_list,
                                tf.argmax(proposal_scores, axis=-1))
    proposal_scores = tf.reduce_max(proposal_scores, axis=-1)

    image = tf.cast(tf.image.resize_images(image, [height, width]), tf.uint8)

    top_k = 5
    top_k_boxes, top_k_scores, top_k_labels = model_utils.get_top_k_boxes_and_scores(
        proposals, tf.sigmoid(proposal_scores), proposal_labels, k=top_k)

    image = plotlib.draw_rectangles(
        image,
        boxes=top_k_boxes,
        scores=top_k_scores,
        labels=top_k_labels,
        color=plotlib.RED)

    tf.summary.image("proposals", image, max_outputs=5)

  def _visl_class_activation_map_list(self,
                                      image,
                                      class_act_map_list,
                                      anchors,
                                      anchor_scores_list,
                                      vocabulary_list,
                                      height=224,
                                      width=224):
    """Visualize class activation map to the tensorboard.

    Args:
      image: A [batch, height, width, channels] float tensor, 
        ranging from 0 to 255.
      class_act_map_list: A list of class activation map, each of them is a
        [batch, height, width, num_classes] float tensor.
      anchors: A [batch, number_of_anchors, 4] float tensor, in normalized
        coordinates.
      anchor_scores_list: A list of [batch, number_of_anchors, num_classes]
        float tensor.
      vocabulary_list: Names of the classes.
      height: Height of the visualized image.
      width: Width of the visualized image.
    """
    options = self._model_proto

    batch, _, _, _ = utils.get_tensor_shape(image)

    # Initialize text labels to attach to images.

    vocabulary_list = [
        tf.tile(tf.expand_dims(w, axis=0), [batch]) for w in vocabulary_list
    ]
    if options.feature_extractor == cam_model_pb2.CAMModel.FRCNN:
      text_list = [
          tf.tile(tf.expand_dims('scale %.2lf' % (s), axis=0), [batch])
          for s in self._input_scales
      ]
    elif options.feature_extractor == cam_model_pb2.CAMModel.SSD:
      text_list = [
          tf.tile(
              tf.expand_dims('layer %s' % (n.split('/')[-1]), axis=0), [batch])
          for n in self._cnn_feature_names
      ]

    merge_v_fn = lambda x: tf.concat(x, axis=1)
    merge_h_fn = lambda x: tf.concat(x, axis=2)

    # Visualize heat map from different resolutions.

    visl_list = []
    for class_act_map, anchor_scores, text in zip(
        class_act_map_list, anchor_scores_list, text_list):

      image_visl = plotlib.draw_caption(
          tf.cast(
              tf.image.resize_images(image, [height * 2, width * 2]), tf.uint8),
          text, (5, 5), plotlib.RED)

      visl_list_at_i = []
      for class_id, x in enumerate(
          tf.unstack(
              tf.image.resize_images(class_act_map, [height, width]), axis=-1)):
        class_name = vocabulary_list[class_id]

        # Look for the top-k box.

        top_k = 1
        top_k_boxes, top_k_scores, _ = model_utils.get_top_k_boxes_and_scores(
            anchors, anchor_scores[:, :, class_id], k=top_k)

        # Draw class-related heat map.

        x = plotlib.convert_to_heatmap(x, normalize=False)
        x = tf.image.convert_image_dtype(x, tf.uint8)
        x = plotlib.draw_caption(x, class_name, org=(0, 0), color=plotlib.RED)

        # Draw bounding box.

        x = plotlib.draw_rectangles(
            x, boxes=top_k_boxes, scores=top_k_scores, color=plotlib.BLACK)
        image_visl = plotlib.draw_rectangles(
            image_visl, boxes=top_k_boxes, color=plotlib.GREEN)

        visl_list_at_i.append(x)

      half_size = len(visl_list_at_i) // 2
      visl_list.append(
          merge_h_fn([image_visl] + [
              merge_v_fn([
                  merge_h_fn(visl_list_at_i[:half_size]),
                  merge_h_fn(visl_list_at_i[half_size:])
              ])
          ]))

    tf.summary.image("image", merge_v_fn(visl_list), max_outputs=5)

  def _extract_class_label(self, num_captions, caption_strings, caption_lengths,
                           vocabulary_list):
    """Encodes labels.

    Args:
      num_captions: a [batch] int tensor, should always be ONE.
      caption_strings: a [batch, num_captions, max_caption_len] string tensor.
      caption_lengths: a [batch, num_captions] int tensor.
      vocabulary_list: a list of words of length `num_classes`.

    Returns:
      class_label: a [batch, num_classes] float tensor.
    """
    with tf.name_scope('extract_class_label'):
      batch, num_captions, max_caption_len = utils.get_tensor_shape(
          caption_strings)

      caption_string = caption_strings[:, 0, :]
      caption_length = caption_lengths[:, 0]

      categorical_col = tf.feature_column.categorical_column_with_vocabulary_list(
          key='name_to_class_id',
          vocabulary_list=vocabulary_list,
          num_oov_buckets=1)
      indicator_col = tf.feature_column.indicator_column(categorical_col)
      indicator = tf.feature_column.input_layer(
          {
              'name_to_class_id': caption_strings
          },
          feature_columns=[indicator_col])
      class_label = tf.cast(indicator[:, :-1] > 0, tf.float32)
      class_label.set_shape([batch, len(vocabulary_list)])

    return class_label

  def _calc_global_pooling(self, class_act_map_list, max_pool=False):
    """Applies global average/maximum pooling to predict the logits.

    Args:
      class_act_map_list: a list of class activation map, each of them is a
        [batch, feature_height, feature_width, num_classes] float tensor.

    Returns:
      first_stage_logits_list: a list of [batch, num_classes] float tensor denoting 
        the logit for each scale.
    """
    with tf.name_scope('calc_global_pooling'):
      batch, _, _, num_classes = utils.get_tensor_shape(class_act_map_list[0])
      if max_pool:
        pool_fn = tf.reduce_max
      else:
        pool_fn = tf.reduce_mean

      first_stage_logits_list = [
          pool_fn(class_act_map, axis=[1, 2])
          for class_act_map in class_act_map_list
      ]
    return first_stage_logits_list

  def _calc_anchor_scores(self, class_act_map, anchors, num_boxes_per_class=1):
    """Calculates class activation box based on the class activation map.

    Args:
      class_act_map: A [batch, height, width, num_classes] float tensor.
      anchor_boxes: A [batch, number_of_anchors, 4] float tensor.

    Returns:
      anchor_scores: A [batch, number_of_anchors, num_classes] tensor.
    """
    with tf.name_scope('calc_anchor_scores'):
      batch, height, width, num_classes = utils.get_tensor_shape(class_act_map)
      ymin, xmin, ymax, xmax = tf.unstack(anchors, axis=-1)
      anchors_abs = tf.stack([
          tf.to_int64(tf.round(ymin * tf.to_float(height))),
          tf.to_int64(tf.round(xmin * tf.to_float(width))),
          tf.to_int64(tf.round(ymax * tf.to_float(height))),
          tf.to_int64(tf.round(xmax * tf.to_float(width)))
      ],
                             axis=-1)

      fn = model_utils.build_proposal_saliency_fn(
          func_name='wei', border_ratio=0.2, purity_weight=1.0)
      anchor_scores = fn(class_act_map, anchors_abs)
    return anchor_scores

  def _calc_proposal_boxes(self, anchors, anchor_scores_list, top_k=1):
    """Calculates proposal boxes.

    Args:
      anchors: A [batch, number_of_anchors, 4] float tensor, in normalized
        coordinates.
      anchor_scores_list: A list of [batch, number_of_anchors, num_classes]
        float tensor.

    Returns:
      number_of_proposals: A [batch] int tensor.
      proposals: a [batch, max_num_proposals, 4] float tensor.
    """
    boxes_list = []
    for anchor_scores in anchor_scores_list:
      for class_id in range(len(self._vocabulary_list)):
        top_k_boxes, top_k_scores, _ = model_utils.get_top_k_boxes_and_scores(
            anchors, anchor_scores[:, :, class_id], k=top_k)
        boxes_list.append(top_k_boxes)
    proposals = tf.concat(boxes_list, axis=1)
    _, number_of_proposals, _ = utils.get_tensor_shape(proposals)
    return number_of_proposals, proposals

  def _calc_proposal_scores(self,
                            image,
                            number_of_proposals,
                            proposals,
                            crop_size=28,
                            num_classes=20):
    """Calculates second stage logits.

    Args:
      image: A [batch, height, width, 3] float tensor.
      number_of_proposals: A [batch] int tensor.
      proposals: a [batch, max_num_proposals, 4] float tensor.

    Returns:
      proposal_scores: a [batch, max_num_proposals, num_classes] float tensor.
    """
    options = self._model_proto
    is_training = self._is_training

    # Stack proposal boxes.

    batch, max_num_proposals, _ = utils.get_tensor_shape(proposals)
    box_ind = tf.expand_dims(tf.range(batch), axis=-1)
    box_ind = tf.reshape(tf.tile(box_ind, [1, max_num_proposals]), [-1])

    proposals_flattened = tf.reshape(proposals, [-1, 4])

    # Crop image and predict using CNN, image_cropped
    #   shape = [ batch * max_num_proposals, crop_size, crop_size, 3].

    image_cropped = tf.image.crop_and_resize(
        image,
        boxes=proposals_flattened,
        box_ind=box_ind,
        crop_size=[crop_size, crop_size])
    preprocessed_image_cropped = model_utils.preprocess_image(
        image_cropped, options.preprocess_method)

    with tf.variable_scope(CAMVariableScopes.cnn, reuse=True):
      feature_cropped = model_utils.vgg_16(
          preprocessed_image_cropped,
          is_training=is_training and options.cnn_trainable)
      feature_cropped = slim.dropout(
          feature_cropped,
          options.cnn_dropout_keep_prob,
          is_training=is_training)

    with tf.variable_scope(CAMVariableScopes.image_proj, reuse=True):
      proposal_scores = self._project_image(
          feature_cropped,
          num_outputs=num_classes,
          hyperparams=options.image_proj_hyperparams,
          is_training=is_training)

    proposal_scores = tf.squeeze(proposal_scores, [1, 2])
    proposal_scores = tf.reshape(proposal_scores,
                                 [batch, max_num_proposals, num_classes])

    # Visualize the crops.

    height = width = 112
    patches = tf.image.resize_images(image_cropped, [height, width])
    patch_labels = tf.reshape(
        tf.gather(self._vocabulary_list, tf.argmax(proposal_scores, axis=-1)),
        [-1, 1])
    patch_scores = tf.reshape(tf.reduce_max(proposal_scores, axis=-1), [-1, 1])

    patches = plotlib.draw_rectangles(
        tf.cast(patches, tf.uint8),
        boxes=tf.tile(
            tf.constant([[[0.1, 0.1, 0.1, 0.1]]]),
            [batch * max_num_proposals, 1, 1]),
        scores=tf.sigmoid(patch_scores),
        labels=patch_labels,
        color=plotlib.RED,
        fontscale=0.6)
    patches = tf.reshape(patches, [batch, max_num_proposals, height, width, 3])
    visl_crops = tf.concat(tf.unstack(patches, axis=1), axis=2)

    tf.summary.image("crops", visl_crops, max_outputs=5)

    return proposal_scores

  def _binarize(self, x, threshold=0.0):
    """Binarizes `x` using a given threshold.
    """
    ones = tf.ones_like(x, dtype=tf.float32)
    zeros = tf.zeros_like(x, dtype=tf.float32)
    return tf.where(x > threshold, x=x, y=zeros)

  def build_prediction(self,
                       examples,
                       prediction_task=CAMTasks.image_label,
                       **kwargs):
    """Builds tf graph for prediction.

    Args:
      examples: dict of input tensors keyed by name.
      prediction_task: the specific prediction task.

    Returns:
      predictions: dict of prediction results keyed by name.
    """
    options = self._model_proto

    # Calculate the class activation map.

    image = examples[InputDataFields.image]
    batch, height, width, _ = utils.get_tensor_shape(image)

    (cnn_feature_map_list,
     class_act_map_list) = self._calc_class_activation_map_list(
         image, num_classes=len(self._vocabulary_list))
    first_stage_logits_list = self._calc_global_pooling(
        class_act_map_list, max_pool=options.max_pool)

    resized_class_act_map_list = [
        tf.image.resize_images(self._binarize(tf.sigmoid(x)), [height, width])
        for x in class_act_map_list
    ]

    # Extract class-related proposal boxes from the activation map.

    anchors = tf.tile(tf.expand_dims(self._anchors, axis=0), [batch, 1, 1])
    anchor_scores_list = [
        self._calc_anchor_scores(x, anchors) for x in resized_class_act_map_list
    ]
    number_of_proposals, proposals = self._calc_proposal_boxes(
        anchors, anchor_scores_list)

    # Calculate the second stage logits.

    proposal_scores = self._calc_proposal_scores(image, number_of_proposals,
                                                 proposals)
    second_stage_logits = tf.reduce_mean(proposal_scores, axis=1)

    # Visualize the boxes.

    self._visl_class_activation_map_list(image, resized_class_act_map_list,
                                         anchors, anchor_scores_list,
                                         self._vocabulary_list)

    self._visl_proposals(image, number_of_proposals, proposals, proposal_scores,
                         self._vocabulary_list)

    prediction_dict = {
        CAMPredictions.class_act_map: resized_class_act_map_list[0],
        CAMPredictions.class_act_map_list: class_act_map_list,
        CAMPredictions.first_stage_logits_list: first_stage_logits_list,
        CAMPredictions.second_stage_logits: second_stage_logits,
        CAMPredictions.proposals: proposals,
        CAMPredictions.proposal_scores: proposal_scores,
    }

    # Extract the class label.

    if prediction_task == CAMTasks.image_label:
      labels = self._extract_class_label(
          num_captions=examples[InputDataFields.num_captions],
          caption_strings=examples[InputDataFields.caption_strings],
          caption_lengths=examples[InputDataFields.caption_lengths],
          vocabulary_list=self._vocabulary_list)
      prediction_dict[CAMPredictions.labels] = labels

    return prediction_dict

  def build_loss(self, predictions, **kwargs):
    """Build tf graph to compute loss.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      loss_dict: dict of loss tensors keyed by name.
    """
    options = self._model_proto

    with tf.name_scope('losses'):

      loss_dict = {}

      labels, first_stage_logits_list, second_stage_logits = (
          predictions[CAMPredictions.labels],
          predictions[CAMPredictions.first_stage_logits_list],
          predictions[CAMPredictions.second_stage_logits])

      # The 1st-stage cross entropy loss.

      for index, logits in enumerate(first_stage_logits_list):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)
        cross_entropy_loss = tf.reduce_mean(losses)
        loss_dict['first_stage_cross_entropy_loss_{}'.format(
            index)] = cross_entropy_loss

      # The 2nd-stage cross entropy loss.
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=second_stage_logits)
      cross_entropy_loss = tf.reduce_mean(losses)
      loss_dict['second_stage_cross_entropy_loss'] = cross_entropy_loss

      return loss_dict

  def build_evaluation(self, predictions, **kwargs):
    """Build tf graph to evaluate the model.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      eval_metric_ops: dict of metric results keyed by name. The values are the
        results of calling a metric function, namely a (metric_tensor, 
        update_op) tuple. see tf.metrics for details.
    """
    with tf.name_scope('losses'):
      labels, first_stage_logits_list, second_stage_logits = (
          predictions[CAMPredictions.labels],
          predictions[CAMPredictions.first_stage_logits_list],
          predictions[CAMPredictions.second_stage_logits])

      metrics = {}
      for index, logits in enumerate(first_stage_logits_list):
        for (label, logit) in zip(
            tf.unstack(labels, axis=0), tf.unstack(logits, axis=0)):
          for top_k in [1, 3]:
            label_indices = tf.squeeze(tf.where(tf.greater(label, 0)), axis=-1)
            map_val, map_update = tf.metrics.average_precision_at_k(
                labels=label_indices, predictions=logit, k=top_k)
            metrics.update({
                'metrics/first_stage_mAP_%i_top%i' % (index, top_k):
                (map_val, map_update)
            })

      for (label, logit) in zip(
          tf.unstack(labels, axis=0), tf.unstack(second_stage_logits, axis=0)):
        for top_k in [1, 3]:
          label_indices = tf.squeeze(tf.where(tf.greater(label, 0)), axis=-1)
          map_val, map_update = tf.metrics.average_precision_at_k(
              labels=label_indices, predictions=logit, k=top_k)
          metrics.update({
              'metrics/second_stage_mAP_%i_top%i' % (index, top_k): (map_val,
                                                                     map_update)
          })
    return metrics
