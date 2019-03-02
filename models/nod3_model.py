from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from models.model_base import ModelBase
from protos import nod3_model_pb2

from nets import nets_factory
from nets import vgg
from core import imgproc
from core import utils
from core import plotlib
from core.standard_fields import InputDataFields
from core.standard_fields import NOD3Predictions
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
  """NOD3 model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of nod3_model_pb2.NOD3Model
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, nod3_model_pb2.NOD3Model):
      raise ValueError('The model_proto has to be an instance of NOD3Model.')

    options = model_proto

    self._open_vocabulary_list = model_utils.read_vocabulary(
        options.open_vocabulary_file)
    with open(options.open_vocabulary_glove_file, 'rb') as fid:
      self._open_vocabulary_initial_embedding = np.load(fid)

    #self._vocabulary_list = model_utils.read_vocabulary(options.vocabulary_file)

    #self._num_classes = len(self._vocabulary_list)

    self._num_classes = options.number_of_classes

    self._feature_extractor = build_faster_rcnn_feature_extractor(
        options.feature_extractor, is_training,
        options.inplace_batchnorm_update)

    self._midn_post_process_fn = function_builder.build_post_processor(
        options.midn_post_process)

    self._oicr_post_process_fn = function_builder.build_post_processor(
        options.oicr_post_process)

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
                               tanh_hiddens=50,
                               name_scope='multi_instance_detection',
                               var_scope='multi_instance_detection'):
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
    with tf.name_scope(name_scope):

      batch, max_num_proposals, _ = utils.get_tensor_shape(proposal_features)
      mask = tf.sequence_mask(
          num_proposals, maxlen=max_num_proposals, dtype=tf.float32)
      mask = tf.expand_dims(mask, axis=-1)

      # Calculates the values of following tensors:
      #   logits_r_given_c shape = [batch, max_num_proposals, num_classes].
      #   logits_c_given_r shape = [batch, max_num_proposals, num_classes].

      with tf.variable_scope(var_scope):
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
                          num_classes=20,
                          name_scope='multi_instance_detection',
                          var_scope='multi_instance_detection'):
    """Builds the Multiple Instance Detection Network.

    MIDN: An attention network.

    Args:
      num_proposals: A [batch] int tensor.
      proposal_features: A [batch, max_num_proposals, features_dims] 
        float tensor.
      num_classes: Number of classes.

    Returns:
      logits: A [batch, num_classes] float tensor.
      proposal_scores: A [batch, max_num_proposals, num_classes] float tensor.
      proba_r_given_c: A [batch, max_num_proposals, num_classes] float tensor.
    """
    with tf.name_scope(name_scope):

      batch, max_num_proposals, _ = utils.get_tensor_shape(proposal_features)
      mask = tf.sequence_mask(
          num_proposals, maxlen=max_num_proposals, dtype=tf.float32)
      mask = tf.expand_dims(mask, axis=-1)

      # Calculates the values of following tensors:
      #   logits_r_given_c shape = [batch, max_num_proposals, num_classes].
      #   logits_c_given_r shape = [batch, max_num_proposals, num_classes].

      with tf.variable_scope(var_scope):
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
      proposal_scores = predictions[NOD3Predictions.oicr_proposal_scores +
                                    '_at_{}'.format(i)]
      proposal_scores = tf.stop_gradient(proposal_scores)
      if i > 0:
        post_process_fn = self._oicr_post_process_fn
        proposal_scores = tf.nn.softmax(proposal_scores, axis=-1)[:, :, 1:]

      # Post process.

      (num_detections, detection_boxes, detection_scores,
       detection_classes) = post_process_fn(proposals, proposal_scores)

      #model_utils.visl_detections(
      #    inputs,
      #    num_detections,
      #    detection_boxes,
      #    detection_scores,
      #    tf.gather(self._vocabulary_list, tf.to_int32(detection_classes - 1)),
      #    name='detection_{}'.format(i))

      results[DetectionResultFields.num_detections +
              '_at_{}'.format(i)] = num_detections
      results[DetectionResultFields.detection_boxes +
              '_at_{}'.format(i)] = detection_boxes
      results[DetectionResultFields.detection_scores +
              '_at_{}'.format(i)] = detection_scores
      results[DetectionResultFields.detection_classes +
              '_at_{}'.format(i)] = detection_classes
    return results

  def _encode_tokens(self,
                     tokens,
                     embedding_dims,
                     vocabulary_list,
                     trainable=True):
    """Encodes tokens to the embedding vectors.

    Args:
      tokens: A list of words or a string tensor of shape [#tokens].
      embedding_dims: Embedding dimensions.
      vocabulary_list: A list of words.

    Returns:
      A [#tokens, embedding_dims] float tensor.
    """
    table = tf.contrib.lookup.index_table_from_tensor(
        vocabulary_list, num_oov_buckets=1)
    token_ids = table.lookup(tokens)

    unk_emb = 0.03 * (np.random.rand(1, embedding_dims) * 2 - 1)
    initial_value = np.concatenate(
        [self._open_vocabulary_initial_embedding, unk_emb], axis=0)

    with tf.variable_scope('word_embedding'):
      embedding_weights = tf.get_variable(
          name='weights',
          initializer=initial_value.astype(np.float32),
          trainable=trainable)
    token_embedding = tf.nn.embedding_lookup(
        embedding_weights, token_ids, max_norm=None)
    tf.summary.histogram('triplet/token_embedding', token_embedding)
    return token_embedding

  def _extract_text_feature(self,
                            text_strings,
                            text_lengths,
                            vocabulary_list,
                            embedding_dims=50,
                            trainable=True):
    """Extracts text feature.

    Args:
      text_strings: A [batch, max_text_length] string tensor.
      text_lengths: A [batch] int tensor.
      vocabulary_list: A list of words.

    Returns:
      text_features: a [batch, max_text_length, feature_dims] float tensor.
    """
    batch, max_text_length = utils.get_tensor_shape(text_strings)

    text_strings_flattented = tf.reshape(text_strings, [-1])
    text_feature_flattened = self._encode_tokens(
        text_strings_flattented, embedding_dims, vocabulary_list, trainable)

    text_feature = tf.reshape(text_feature_flattened,
                              [batch, max_text_length, embedding_dims])
    return text_feature

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

    # Gather image and proposals.

    (inputs, num_proposals,
     proposals) = (examples[InputDataFields.image],
                   examples[InputDataFields.num_proposals],
                   examples[InputDataFields.proposals])

    tf.summary.image('inputs', inputs, max_outputs=10)
    model_utils.visl_proposals(
        inputs, num_proposals, proposals, name='proposals', top_k=100)

    # Gather in-batch captions.

    (image_id, num_captions, caption_strings,
     caption_lengths) = (examples[InputDataFields.image_id],
                         examples[InputDataFields.num_captions],
                         examples[InputDataFields.caption_strings],
                         examples[InputDataFields.caption_lengths])
    image_id = tf.string_to_number(image_id, out_type=tf.int64)

    (image_ids_gathered, caption_strings_gathered,
     caption_lengths_gathered) = model_utils.gather_in_batch_captions(
         image_id, num_captions, caption_strings, caption_lengths)

    # Word embedding

    caption_features_gathered = self._extract_text_feature(
        caption_strings_gathered,
        caption_lengths_gathered,
        vocabulary_list=self._open_vocabulary_list,
        embedding_dims=options.embedding_dims,
        trainable=options.train_word_embedding)

    # FRCNN.

    proposal_features = self._extract_frcnn_feature(inputs, num_proposals,
                                                    proposals)

    # Build MIDN network, for both image and text.
    #   class_logits shape = [batch, num_classes]
    #   proposal_scores shape = [batch, max_num_proposals, num_classes].
    #   proba_r_given_c shape = [batch, max_num_proposals, num_classes].

    assert options.attention_type == nod3_model_pb2.NOD3Model.PER_CLASS

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      (midn_class_logits, midn_proposal_scores,
       midn_proba_r_given_c) = self._build_midn_network(
           num_proposals,
           proposal_features,
           num_classes=self._num_classes,
           name_scope='image_midn',
           var_scope='image_midn')

    with slim.arg_scope(
        build_hyperparams(options.text_fc_hyperparams, is_training)):
      (text_class_logits, text_proposal_scores,
       text_proba_r_given_c) = self._build_midn_network_tanh(
           caption_lengths_gathered,
           caption_features_gathered,
           num_classes=self._num_classes,
           tanh_hiddens=options.tanh_hiddens,
           name_scope='text_midn',
           var_scope='text_midn')

    # Extract token statistics.

    tokens = tf.constant(self._open_vocabulary_list)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      token_embeddings = self._encode_tokens(
          tokens=tokens,
          embedding_dims=options.embedding_dims,
          vocabulary_list=self._open_vocabulary_list,
          trainable=options.train_word_embedding)

#    with slim.arg_scope(
#        build_hyperparams(options.text_fc_hyperparams, is_training)):
#      with tf.variable_scope('text_midn', reuse=True):
#        token_logits_r_given_c = slim.fully_connected(
#            token_embeddings,
#            num_outputs=self._num_classes,
#            activation_fn=None,
#            scope='proba_r_given_c')
#        token_logits_c_given_r = slim.fully_connected(
#            token_embeddings,
#            num_outputs=self._num_classes,
#            activation_fn=None,
#            scope='proba_c_given_r')
#        #per_class_token_scores = tf.multiply(
#        #    tf.nn.sigmoid(token_logits_c_given_r),
#        #    tf.nn.softmax(token_logits_r_given_c, axis=0))
#        per_class_token_scores = token_logits_r_given_c

    # Compute similarity.

    tf.summary.histogram('triplet/text_logits', text_class_logits)
    tf.summary.histogram('triplet/image_logits', midn_class_logits)

    with tf.name_scope('calc_cross_modal_similarity'):
      text_class_proba = tf.nn.sigmoid(text_class_logits)
      midn_class_proba = tf.nn.sigmoid(midn_class_logits)

      similarity = model_utils.calc_pairwise_similarity(
          feature_a=midn_class_proba,
          feature_b=text_class_proba,
          l2_normalize=True,
          dropout_keep_prob=options.cross_modal_dropout_keep_prob,
          is_training=is_training)

    tf.summary.histogram('triplet/text_proba', text_class_proba)
    tf.summary.histogram('triplet/image_proba', midn_class_proba)

    predictions = {
        DetectionResultFields.class_labels:
        tf.constant([str(i) for i in range(self._num_classes)]),
        DetectionResultFields.num_proposals:
        num_proposals,
        DetectionResultFields.proposal_boxes:
        proposals,
        NOD3Predictions.image_id:
        image_id,
        NOD3Predictions.image_ids_gathered:
        image_ids_gathered,
        NOD3Predictions.similarity:
        similarity,
        NOD3Predictions.tokens:
        tokens,
        NOD3Predictions.per_class_token_scores:
        per_class_token_scores,
    }

    # Build the OICR network.
    #   proposal_scores shape = [batch, max_num_proposals, 1 + num_classes].
    #   See `Multiple Instance Detection Network with OICR`.

    predictions[NOD3Predictions.oicr_proposal_scores +
                '_at_0'] = midn_proposal_scores

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      for i in range(options.oicr_iterations):
        predictions[NOD3Predictions.oicr_proposal_scores +
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
        proposals_scores = predictions[NOD3Predictions.oicr_proposal_scores +
                                       '_at_{}'.format(i)]
        proposal_scores_list[i].append(proposals_scores)

      reuse = True

    # Aggregate (averaging) predictions from different resolutions.

    predictions_aggregated = predictions
    for i in range(1 + options.oicr_iterations):
      proposal_scores = tf.stack(proposal_scores_list[i], axis=-1)
      proposal_scores = tf.reduce_mean(proposal_scores, axis=-1)
      predictions_aggregated[NOD3Predictions.oicr_proposal_scores +
                             '_at_{}'.format(i)] = proposal_scores

    predictions_aggregated.update(
        self._post_process(inputs, predictions_aggregated))

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

    # Extracts tensors and shapes.

    (image_id, image_ids_gathered,
     similarity) = (predictions[NOD3Predictions.image_id],
                    predictions[NOD3Predictions.image_ids_gathered],
                    predictions[NOD3Predictions.similarity])

    # Triplet loss.
    # Distance matrix, shape = [batch, num_captions_in_batch].

    distance = 1.0 - similarity
    pos_mask = tf.cast(
        tf.equal(
            tf.expand_dims(image_id, axis=1),
            tf.expand_dims(image_ids_gathered, axis=0)), tf.float32)
    neg_mask = 1.0 - pos_mask
    distance_ap = utils.masked_maximum(distance, pos_mask)

    if options.triplet_loss_use_semihard:

      # Use the semihard.

      # negatives_outside: smallest D_an where D_an > D_ap.

      mask = tf.cast(tf.greater(distance, distance_ap), tf.float32)
      mask = mask * neg_mask
      negatives_outside = utils.masked_minimum(distance, mask)

      # negatives_inside: largest D_an.

      negatives_inside = utils.masked_maximum(distance, neg_mask)

      # distance_an: the semihard negatives.

      mask_condition = tf.greater(
          tf.reduce_sum(mask, axis=1, keepdims=True), 0.0)

      distance_an = tf.where(mask_condition, negatives_outside,
                             negatives_inside)

    else:

      # Use the hardest.

      distance_an = utils.masked_minimum(distance, neg_mask)

    losses = tf.maximum(distance_ap - distance_an + options.triplet_loss_margin,
                        0)

    num_loss_examples = tf.count_nonzero(losses, dtype=tf.float32)
    #loss = tf.div(
    #    tf.reduce_sum(losses),
    #    _EPSILON + num_loss_examples,
    #    name="triplet_loss")
    loss = tf.reduce_mean(losses)

    tf.summary.scalar('loss/num_loss_examples', num_loss_examples)
    tf.summary.scalar('loss/triplet_loss', loss)

    return {'triplet_loss': loss}

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
