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

slim = tf.contrib.slim
_EPSILON = 1e-8


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

    self._open_vocabulary_list = model_utils.read_vocabulary(
        options.open_vocabulary_file)
    with open(options.open_vocabulary_glove_file, 'rb') as fid:
      self._open_vocabulary_initial_embedding = np.load(fid)

    self._vocabulary_list = model_utils.read_vocabulary(options.vocabulary_file)
    self._num_classes = len(self._vocabulary_list)
    self._midn_post_process_fn = function_builder.build_post_processor(
        options.midn_post_process)
    self._oicr_post_process_fn = function_builder.build_post_processor(
        options.oicr_post_process)

    if options.HasField('synonyms_file'):
      self._synonyms = model_utils.read_synonyms(options.synonyms_file)

  def _extract_exact_match_label(self, class_texts, vocabulary_list):
    """Extracts class labels, used for EXACT MATCHING.

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
      indicator = tf.feature_column.input_layer({
          'name_to_id': class_texts
      },
                                                feature_columns=[indicator_col])
      labels = tf.cast(indicator[:, :-1] > 0, tf.float32)
      labels.set_shape([batch, len(vocabulary_list)])

    return labels

  def _encode_tokens(self,
                     tokens,
                     embedding_dims,
                     vocabulary_list,
                     initial_embedding=None,
                     trainable=True,
                     init_width=0.03,
                     max_norm=None):
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

    if initial_embedding is not None:
      tf.logging.info('Word embedding is initialized from numpy array.')
      unk_emb = init_width * (np.random.rand(1, embedding_dims) * 2 - 1)
      initial_value = np.concatenate([initial_embedding, unk_emb], axis=0)
    else:
      tf.logging.warn('Word embedding is not initialized!')
      initial_value = init_width * (
          np.random.rand(1 + len(vocabulary_list), embedding_dims) * 2 - 1)

    embedding_weights = tf.get_variable(
        name='weights',
        initializer=initial_value.astype(np.float32),
        trainable=trainable)
    token_embedding = tf.nn.embedding_lookup(
        embedding_weights, token_ids, max_norm=max_norm)
    tf.summary.histogram('token_embedding', token_embedding)
    return token_ids, token_embedding

  def _extract_extend_match_label(self,
                                  class_texts,
                                  vocabulary_dict,
                                  num_classes=80):
    """Extracts class labels, used for EXTEND MATCHING.

    Args:
      class_texts: a [batch, max_num_objects] string tensor.
      vocabulary_list: a list of words of length `num_classes`.

    Returns:
      labels: a [batch, num_classes] float tensor.
    """
    with tf.name_scope('extract_class_label_v2'):
      batch, _ = utils.get_tensor_shape(class_texts)

      oov = num_classes
      items = vocabulary_dict.items()
      keys = [k for k, v in items]
      values = [v for k, v in items]
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(keys, values), oov)

      ids = table.lookup(class_texts)
      labels = tf.one_hot(indices=ids, depth=oov + 1, dtype=tf.float32)
      labels = tf.reduce_max(labels, axis=1)[:, :-1]

    return labels

  def _extract_text_feature(self,
                            text_strings,
                            text_lengths,
                            vocabulary_list,
                            initial_embedding=None,
                            embedding_dims=50,
                            trainable=True,
                            max_norm=None):
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
    token_ids_flatterned, text_features_flattened = self._encode_tokens(
        text_strings_flattented,
        embedding_dims,
        vocabulary_list,
        initial_embedding,
        trainable,
        max_norm=max_norm)

    token_ids = tf.reshape(token_ids_flatterned, [batch, max_text_length])
    text_features = tf.reshape(text_features_flattened,
                               [batch, max_text_length, embedding_dims])
    return token_ids, text_features

  def _extract_w2v_match_label(self,
                               texts,
                               vocabulary_list,
                               open_vocabulary_list,
                               embedding_dims=50):
    """Extracts class labels.

    Args:
      texts: a [batch, max_text_length] string tensor.
      vocabulary_list: a list of words of length `num_classes`.
      open_vocabulary_list: a list of words of length `num_tokens`.

    Returns:
      labels: a [batch, num_classes] float tensor.
    """
    for cls in vocabulary_list:
      if not cls in set(open_vocabulary_list):
        tf.logging.warn('Unknown class name {}'.format(cls))

    # Class embedding shape = [num_classes, embedding_dims].
    # Text embedding shape = [batch, max_text_length, embedding_dims].

    with tf.variable_scope('token_embedding'):
      class_token_ids, class_embeddings = self._encode_tokens(
          tokens=tf.constant(vocabulary_list),
          initial_embedding=self._open_vocabulary_initial_embedding,
          embedding_dims=embedding_dims,
          vocabulary_list=open_vocabulary_list,
          trainable=False,
          max_norm=None)

    with tf.variable_scope('token_embedding', reuse=True):
      text_token_ids, text_embeddings = self._extract_text_feature(
          texts,
          None,
          open_vocabulary_list,
          initial_embedding=self._open_vocabulary_initial_embedding,
          embedding_dims=embedding_dims,
          trainable=False,
          max_norm=None)

    # Compute text-to-class similarity.

    class_embeddings = tf.nn.l2_normalize(class_embeddings, axis=-1)
    text_embeddings = tf.nn.l2_normalize(text_embeddings, axis=-1)

    dot_product = tf.multiply(
        tf.expand_dims(tf.expand_dims(class_embeddings, axis=0), axis=0),
        tf.expand_dims(text_embeddings, axis=2))
    similarity = tf.reduce_sum(dot_product, axis=-1)

    oov = len(open_vocabulary_list)
    mask = tf.to_float(tf.not_equal(text_token_ids, oov))
    similarity_aggr = utils.masked_maximum(
        data=similarity, mask=tf.expand_dims(mask, axis=-1), dim=1)
    similarity_aggr = tf.squeeze(similarity_aggr, axis=1)

    batch, num_classes = utils.get_tensor_shape(similarity_aggr)
    indices0 = tf.range(batch, dtype=tf.int64)
    indices1 = tf.argmax(similarity_aggr, axis=1)
    indices = tf.stack([indices0, indices1], axis=-1)

    labels = tf.sparse_to_dense(
        indices, output_shape=[batch, num_classes], sparse_values=1.0)
    return labels

  def _extract_text_classifier_soft_label(self, texts):
    # Careful trainable=False, is_training=False!!!!
    options = self._model_proto
    text_token_ids, text_features = self._extract_text_feature(
        texts,
        None,
        vocabulary_list=self._open_vocabulary_list,
        initial_embedding=self._open_vocabulary_initial_embedding,
        embedding_dims=options.embedding_dims,
        trainable=False,
        max_norm=None)

    with tf.variable_scope('text_classifier'):
      #with slim.arg_scope(
      #    build_hyperparams(options.text_fc_hyperparams, is_training=False)):
      text_features = slim.fully_connected(
          text_features,
          num_outputs=options.text_hidden_units,
          activation_fn=None,
          trainable=False,
          scope='layer1')

      oov = len(self._open_vocabulary_list)
      text_masks = tf.to_float(tf.logical_not(tf.equal(text_token_ids, oov)))

      text_features = utils.masked_maximum(
          data=text_features, mask=tf.expand_dims(text_masks, axis=-1), dim=1)
      text_features = tf.squeeze(text_features, axis=1)
      text_features = tf.nn.relu(text_features)
      logits = slim.fully_connected(
          text_features,
          num_outputs=self._num_classes,
          activation_fn=None,
          trainable=False,
          scope='layer2')

    tf.train.init_from_checkpoint(
        options.text_classifier_checkpoint_path,
        assignment_map={"text_classifier/": "text_classifier/"})
    return tf.stop_gradient(logits)


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
      proposal_scores = predictions[Cap2DetPredictions.oicr_proposal_scores +
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
    predictions = {}
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

    proposal_features = model_utils.extract_frcnn_feature(
        inputs, num_proposals, proposals, options.frcnn_options, is_training)

    # Build MIDN network.
    #   proba_r_given_c shape = [batch, max_num_proposals, num_classes].

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      (midn_class_logits, midn_proposal_scores,
       midn_proba_r_given_c) = self._build_midn_network(
           num_proposals, proposal_features, num_classes=self._num_classes)

    # Build the OICR network.
    #   proposal_scores shape = [batch, max_num_proposals, 1 + num_classes].
    #   See `Multiple Instance Detection Network with OICR`.

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      for i in range(options.oicr_iterations):
        predictions[Cap2DetPredictions.oicr_proposal_scores + '_at_{}'.format(
            i + 1)] = proposal_scores = slim.fully_connected(
                proposal_features,
                num_outputs=1 + self._num_classes,
                activation_fn=None,
                scope='oicr/iter{}'.format(i + 1))

    # Set the predictions.

    predictions.update({
        DetectionResultFields.class_labels:
        tf.constant(self._vocabulary_list),
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
        proposals_scores = predictions[Cap2DetPredictions.oicr_proposal_scores +
                                       '_at_{}'.format(i)]
        proposal_scores_list[i].append(proposals_scores)

      reuse = True

    # Aggregate (averaging) predictions from different resolutions.

    predictions_aggregated = predictions
    for i in range(1 + options.oicr_iterations):
      proposal_scores = tf.stack(proposal_scores_list[i], axis=-1)
      proposal_scores = tf.reduce_mean(proposal_scores, axis=-1)
      predictions_aggregated[Cap2DetPredictions.oicr_proposal_scores +
                             '_at_{}'.format(i)] = proposal_scores

    predictions_aggregated.update(
        self._post_process(inputs, predictions_aggregated))

    return predictions_aggregated

  def _extract_labels(self, examples):
    """Extracts labels based on options.

    Returns:
      a [batch, num_classes] float tensor.
    """
    options = self._model_proto

    # Using ground-truth.

    if options.label_option == cap2det_model_pb2.Cap2DetModel.GROUNDTRUTH:
      return self._extract_exact_match_label(
          class_texts=examples[InputDataFields.object_texts],
          vocabulary_list=self._vocabulary_list)

    # Using exact-matching labels.

    if options.label_option == cap2det_model_pb2.Cap2DetModel.EXACT_MATCH:
      labels_gt = self._extract_exact_match_label(
          class_texts=slim.flatten(examples[InputDataFields.caption_strings]),
          vocabulary_list=model_utils.substitute_class_names(
              self._vocabulary_list))
      examples['debug_groundtruth_labels'] = labels_gt
      return labels_gt

    # Using extended matching.
    if options.label_option == cap2det_model_pb2.Cap2DetModel.EXTEND_MATCH:
      cat_to_id = dict([(x, i) for i, x in enumerate(self._vocabulary_list)])
      for syno, orig in self._synonyms.items():
        cat_to_id[syno] = cat_to_id[orig]
      for orig, syno in model_utils.class_synonyms.items():
        if orig in cat_to_id:
          cat_to_id[syno] = cat_to_id[orig]
      labels_gt = self._extract_extend_match_label(
          class_texts=slim.flatten(examples[InputDataFields.caption_strings]),
          vocabulary_dict=cat_to_id,
          num_classes=self._num_classes)
      examples['debug_groundtruth_labels'] = labels_gt
      return labels_gt

    # Using w2v-matchiong.

    if options.label_option == cap2det_model_pb2.Cap2DetModel.W2V_MATCH:
      labels_gt = self._extract_exact_match_label(
          class_texts=slim.flatten(examples[InputDataFields.caption_strings]),
          vocabulary_list=model_utils.substitute_class_names(
              self._vocabulary_list))
      labels_ps = self._extract_w2v_match_label(
          texts=slim.flatten(examples[InputDataFields.caption_strings]),
          vocabulary_list=model_utils.substitute_class_names(
              self._vocabulary_list),
          open_vocabulary_list=self._open_vocabulary_list,
          embedding_dims=options.embedding_dims)
      examples['debug_groundtruth_labels'] = labels_gt
      examples['debug_pseudo_labels'] = labels_ps

      select_op = tf.reduce_any(labels_gt > 0, axis=-1)
      labels = tf.where(select_op, labels_gt, labels_ps)
      return labels

    # Using exact-matching + text clsr.

    if options.label_option == cap2det_model_pb2.Cap2DetModel.TEXT_CLSF:
      labels_gt = self._extract_exact_match_label(
          class_texts=slim.flatten(examples[InputDataFields.caption_strings]),
          vocabulary_list=model_utils.substitute_class_names(
              self._vocabulary_list))

      ps_logits = self._extract_text_classifier_soft_label(
          texts=slim.flatten(examples[InputDataFields.caption_strings]))
      ps_probas = tf.sigmoid(ps_logits)
      labels_ps = tf.to_float(ps_probas > options.soft_label_threshold)

      select_op = tf.reduce_any(labels_gt > 0, axis=-1)
      labels = tf.where(select_op, labels_gt, labels_ps)
      return labels
    
    raise ValueError('not implemented')

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

      labels = self._extract_labels(examples)

      # Set the label to the InputDataFields.
      examples[InputDataFields.pseudo_groundtruth_prediction] = labels

      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels,
          logits=predictions[Cap2DetPredictions.midn_class_logits])
      loss_dict['midn_cross_entropy_loss'] = tf.multiply(
          tf.reduce_mean(losses), options.midn_loss_weight)

      # Losses of the online instance classifier refinement network.

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

      global_step = tf.train.get_or_create_global_step()
      # oicr_loss_mask = tf.cast(global_step > options.oicr_start_step,
      #                          tf.float32)
      oicr_start_step = max(1, options.oicr_start_step)
      oicr_loss_mask = tf.where(
          global_step > options.oicr_start_step, 1.0,
          tf.div(tf.to_float(global_step), oicr_start_step))

      for i in range(options.oicr_iterations):
        proposal_scores_1 = predictions[Cap2DetPredictions.oicr_proposal_scores
                                        + '_at_{}'.format(i + 1)]
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

    tf.summary.scalar('loss/oicr_mask', oicr_loss_mask)
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
