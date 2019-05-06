from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf

from models.model_base import ModelBase
from protos import advise_gcn_model_v4_pb2

from core import utils
from core.training_utils import build_hyperparams
from models import utils as model_utils
from reader.advise_reader import InputDataFields

from models.registry import register_model_class

slim = tf.contrib.slim
_EPSILON = 1e-8

_FIELD_IMAGE_ID = 'image_id'
_FIELD_IMAGE_IDS_GATHERED = 'image_ids_gathered'
_FIELD_SIMILARITY = 'similarity'
_FIELD_ADJACENCY = 'adjacency'
_FIELD_ADJACENCY_LOGITS = 'adjacency_logits'

_INIT_WIDTH = 0.03


class Model(ModelBase):
  """ADVISE model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of advise_gcn_model_v4_pb2.AdViSEGCN
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, advise_gcn_model_v4_pb2.AdViSEGCNV4Model):
      raise ValueError('The model_proto has to be an instance of AdViSEGCN.')

    options = model_proto

    # Read vocabulary.

    stmt_vocab_with_freq = model_utils.read_vocabulary_with_frequency(
        options.stmt_vocab_list_path)
    self._stmt_vocab_list = [
        word for word, freq in stmt_vocab_with_freq if freq > 5
    ]
    tf.logging.info('Vocab of stmt, len=%i', len(self._stmt_vocab_list))
    slgn_vocab_with_freq = model_utils.read_vocabulary_with_frequency(
        options.slgn_vocab_list_path)
    self._slgn_vocab_list = [
        word for word, freq in slgn_vocab_with_freq if freq > 20
    ]
    tf.logging.info('Vocab of slgn, len=%i', len(self._slgn_vocab_list))
    slgn_kb_vocab_with_freq = model_utils.read_vocabulary_with_frequency(
        options.slgn_kb_vocab_list_path)
    self._slgn_kb_vocab_list = [
        word for word, freq in slgn_kb_vocab_with_freq if freq > 20
    ]
    tf.logging.info('Vocab of slgn kb, len=%i', len(self._slgn_kb_vocab_list))

  def get_scaffold(self):
    """Returns scaffold object used to initialize variables.

    Returns:
      a tf.train.Scaffold instance or None by default.
    """
    options = self._model_proto

    def _init_fn(unused_scaffold, sess):
      """Function for initialization.

      Args:
        sess: a tf.Session instance.
      """
      word2vec, _ = model_utils.load_glove_data(options.glove_path)

      dims = word2vec['the'].shape[0]
      assert dims == options.embedding_dims

      for scope, vocab_list in [('stmt_word_embedding', self._stmt_vocab_list),
                                ('slgn_word_embedding', self._slgn_vocab_list),
                                ('slgn_kb_word_embedding', self._slgn_kb_vocab_list) ]:
        word_embedding = [
            word2vec.get(word, _INIT_WIDTH * (np.random.rand(dims) * 2.0 - 1.0))
            for word in vocab_list
        ]
        word_embedding = np.stack(word_embedding, axis=0)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
          var = tf.get_variable(
              name='{}/weights'.format(scope),
              shape=[len(vocab_list), dims],
              trainable=True)
          var.load(word_embedding, sess)

      tf.logging.info("Initialize using scaffold.")

    scaffold = tf.train.Scaffold(init_fn=_init_fn)
    return scaffold

  def _mask_groundtruth(self, groundtruth_strings, question_strings):
    """Gets groundtruth mask from groundtruth_strings and question_strings.

    Args:
      groundtruth_strings: A [batch_groundtruth, max_groundtruth_text_len] string tensor.
      question_strings: A [batch_question, max_question_text_len] string tensor.

    Returns:
      groundtruth_mask: A [batch_question] boolean tensor, in which `True` 
        denotes the option is correct.
    """
    with tf.name_scope('mask_groundtruth_op'):
      groundtruth_strings = tf.string_strip(
          tf.reduce_join(groundtruth_strings, axis=-1, separator=' '))
      question_strings = tf.string_strip(
          tf.reduce_join(question_strings, axis=-1, separator=' '))
      equal_mat = tf.equal(
          tf.expand_dims(question_strings, axis=1),
          tf.expand_dims(groundtruth_strings, axis=0))
      return tf.reduce_any(equal_mat, axis=-1)

  def _word_embedding(self,
                      stmt_vocab_list,
                      embedding_dims=200,
                      init_width=_INIT_WIDTH,
                      scope='word_embedding'):
    """Gets the word embedding.

    Args:
      stmt_vocab_list: A list of string.

    Returns:
      word_embedding: A tensor of shape [1 + number_of_tokens, dims].
    """
    word_embedding = tf.get_variable(
        name='{}/weights'.format(scope),
        initializer=tf.initializers.random_uniform(
            minval=-init_width, maxval=init_width),
        shape=[len(stmt_vocab_list), embedding_dims],
        trainable=True)
    oov_embedding = tf.get_variable(
        name='{}/oov_weights'.format(scope),
        initializer=tf.initializers.random_uniform(
            minval=-init_width, maxval=init_width),
        shape=[1, embedding_dims],
        trainable=True)
    word_embedding = tf.concat([word_embedding, oov_embedding], axis=0)
    return word_embedding

  def _pairwise_transform(self, feature1, feature2, dims=100, scope=None):
    """Builds attention model to compute logits.

    Args:
      feature1: A [batch, max_node_num1, dims1] float tensor.
      feature2: A [batch, max_node_num2, dims2] float tensor.

    Returns:
      feature_2to1 shape = [batch, max_node_num1, max_node_num2]
    """
    with tf.variable_scope(scope):
      feature1 = tf.contrib.layers.fully_connected(
          inputs=feature1,
          num_outputs=dims,
          activation_fn=None,
          scope='{}/project1'.format(scope))
      feature2 = tf.contrib.layers.fully_connected(
          inputs=feature2,
          num_outputs=dims,
          activation_fn=None,
          scope='{}/project2'.format(scope))
      dot_product = tf.matmul(feature1, feature2, transpose_b=True)
      dot_product = tf.div(dot_product, np.sqrt(dims))
    return dot_product

  def _node_embedding(self, feature_list, batch_norm=False, is_training=False):
    """Gathers node embeddings.
    
    Args:
      feature_list: A list of [batch_i, max_node_num, embedding_dims] float tensors.

    Returns:
      feature: A [batch_i, max_node-num, embedding_dims] float tensor.
      mask: A [batch_i, max_node_num] boolean tensors.
    """
    if batch_norm:
      feature_list = [
          slim.batch_norm(
              x,
              activation_fn=None,
              center=True,
              scale=False,
              is_training=is_training) for x in feature_list
      ]
    for i, feature in enumerate(feature_list):
      tf.summary.histogram('gcn/feature_{}'.format(i + 1), feature)

    return tf.concat(feature_list, axis=1)

  def _adjacency_matrix(self,
                        feature1,
                        mask1,
                        feature2,
                        mask2,
                        feature3,
                        mask3,
                        inference_dims=100,
                        connect_roi_to_slogan=False,
                        scope=None):
    """Computes adjacency matrix.

    Args:
      feature1: A [batch_i, max_node_num1, embedding_dims] float tensor.
      mask1: A [batch, max_node_num1] boolean tensor.
      feature2: A [batch_i, max_node_num2, embedding_dims] float tensor.
      mask2: A [batch, max_node_num2] boolean tensor.
      inference_dims: Dimensions for the inference projection layer.
      connect_roi_to_slogan: If True, connect rois to slogan.

    Returns:
      adjacency: A [batch_i, max_node_num1 + max_node_num2, max_node_num1 + max_node_num2] float tensor.
    """
    batch1, max_node_num1, dims1 = utils.get_tensor_shape(feature1)
    batch2, max_node_num2, dims2 = utils.get_tensor_shape(feature2)
    batch3, max_node_num3, dims3 = utils.get_tensor_shape(feature3)
    assert batch1 == batch2 == batch3

    batch = batch1

    with tf.variable_scope(scope):
      feature_1to1 = self._pairwise_transform(
          feature1, feature1, dims=inference_dims, scope='feature_1to1')
      #feature_1to2 = self._pairwise_transform(
      #    feature2, feature1, dims=inference_dims, scope='feature_1to2')
      feature_2to1 = self._pairwise_transform(
          feature1, feature2, dims=inference_dims, scope='feature_2to1')
      #feature_2to2 = self._pairwise_transform(
      #    feature2, feature2, dims=inference_dims, scope='feature_2to2')

      feature_3to1 = self._pairwise_transform(
          feature1, feature3, dims=inference_dims, scope='feature_3to1')

    tf.summary.histogram('gcn/feature1to1', feature_1to1)
    #tf.summary.histogram('gcn/feature1to2', feature_1to2)
    tf.summary.histogram('gcn/feature2to1', feature_2to1)
    #tf.summary.histogram('gcn/feature2to2', feature_2to2)
    tf.summary.histogram('gcn/feature3to1', feature_3to1)

    # Compute adjacency matrix
    #   shape = [batch, max_node_num1 + max_node_num2, max_node_num1 + max_node_num2]

    logits_to1 = tf.concat([feature_1to1, feature_2to1, feature_3to1], axis=2)
    masks_to1 = tf.expand_dims(tf.concat([mask1, mask2, mask3], axis=1), axis=1)
    adjacency_to1 = utils.masked_softmax(data=logits_to1, mask=masks_to1, dim=2)
    adjacency_to1 = tf.multiply(adjacency_to1, tf.expand_dims(mask1, axis=2))

    if connect_roi_to_slogan:
      assert False
      logits_to2 = tf.concat([feature_1to2, feature_2to2], axis=2)
      masks_to2 = tf.concat([
          tf.tile(tf.expand_dims(mask1, axis=1), [1, max_node_num2, 1]),
          tf.eye(num_rows=max_node_num2, batch_shape=[batch2])
      ],
                            axis=2)
      adjacency_to2 = utils.masked_softmax(
          data=logits_to2, mask=masks_to2, dim=2)
      adjacency_to2 = tf.multiply(adjacency_to2, tf.expand_dims(mask2, axis=2))
    else:
      adjacency_to2 = logits_to2 = tf.fill(
          dims=[
              batch, max_node_num2,
              max_node_num1 + max_node_num2 + max_node_num3
          ],
          value=0.0)

    adjacency_to3 = logits_to3 = tf.fill(
        dims=[
            batch, max_node_num3,
            max_node_num1 + max_node_num2 + max_node_num3
        ],
        value=0.0)

    adjacency_logits = tf.concat([logits_to1, logits_to2, logits_to3], axis=1)
    adjacency = tf.concat([adjacency_to1, adjacency_to2, adjacency_to3], axis=1)

    tf.summary.histogram('gcn/adjacency_logits', adjacency_logits)
    tf.summary.histogram('gcn/adjacency_probas', adjacency)

    return adjacency, adjacency_logits

  def build_prediction(self, examples, **kwargs):
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

    # Decode data fields.

    (image_id, image_feature, roi_feature,
     roi_num) = (examples[InputDataFields.image_id],
                 examples[InputDataFields.image_feature],
                 examples[InputDataFields.roi_feature],
                 examples[InputDataFields.roi_num])
    (slogan_num, slogan_text_string,
     slogan_text_length) = (examples[InputDataFields.slogan_num],
                            examples[InputDataFields.slogan_text_string],
                            examples[InputDataFields.slogan_text_length])
    (slogan_kb_num, slogan_kb_text_string,
     slogan_kb_text_length) = (examples[InputDataFields.slogan_kb_num],
                               examples[InputDataFields.slogan_kb_text_string],
                               examples[InputDataFields.slogan_kb_text_length])
    (groundtruth_num, groundtruth_text_string, groundtruth_text_length) = (
        examples[InputDataFields.groundtruth_num],
        examples[InputDataFields.groundtruth_text_string],
        examples[InputDataFields.groundtruth_text_length])

    if options.add_image_as_a_roi:
      roi_num = tf.add(roi_num, 1)
      roi_feature = tf.concat(
          [tf.expand_dims(image_feature, axis=1), roi_feature], axis=1)

    # Generate in-batch question list.

    if is_training:

      # Sample in-batch negatives for the training mode.

      (image_ids_gathered, stmt_text_string,
       stmt_text_length) = model_utils.gather_in_batch_captions(
           image_id, groundtruth_num, groundtruth_text_string,
           groundtruth_text_length)
    else:

      # Use the negatives in the tf.train.Example for the evaluation mode.

      batch = image_id.get_shape()[0].value
      assert batch == 1, "We only support `batch == 1` for evaluation."

      (question_num, question_text_string,
       question_text_length) = (examples[InputDataFields.question_num],
                                examples[InputDataFields.question_text_string],
                                examples[InputDataFields.question_text_length])

      stmt_text_string = question_text_string[0][:question_num[0], :]
      stmt_text_length = question_text_length[0][:question_num[0]]

      stmt_mask = self._mask_groundtruth(
          groundtruth_strings=groundtruth_text_string[0]
          [:groundtruth_num[0], :],
          question_strings=question_text_string[0][:question_num[0], :])

      image_ids_gathered = tf.where(
          stmt_mask,
          x=tf.fill(tf.shape(stmt_mask), image_id[0]),
          y=tf.fill(tf.shape(stmt_mask), tf.constant(-1, dtype=tf.int64)))

    # Word embedding processes.

    stmt_text_feature = tf.nn.embedding_lookup(
        params=self._word_embedding(
            self._stmt_vocab_list,
            options.embedding_dims,
            init_width=_INIT_WIDTH,
            scope='stmt_word_embedding'),
        ids=tf.contrib.lookup.index_table_from_tensor(
            self._stmt_vocab_list, num_oov_buckets=1).lookup(stmt_text_string),
        max_norm=None)

    slogan_text_feature = tf.nn.embedding_lookup(
        params=self._word_embedding(
            self._slgn_vocab_list,
            options.embedding_dims,
            init_width=_INIT_WIDTH,
            scope='slgn_word_embedding'),
        ids=tf.contrib.lookup.index_table_from_tensor(
            self._slgn_vocab_list,
            num_oov_buckets=1).lookup(slogan_text_string),
        max_norm=None)
    slogan_kb_text_feature = tf.nn.embedding_lookup(
        params=self._word_embedding(
            self._slgn_kb_vocab_list,
            options.embedding_dims,
            init_width=_INIT_WIDTH,
            scope='slgn_kb_word_embedding'),
        ids=tf.contrib.lookup.index_table_from_tensor(
            self._slgn_kb_vocab_list,
            num_oov_buckets=1).lookup(slogan_kb_text_string),
        max_norm=None)

    # Image representation.
    #   roi_num shape = [batch_i]
    #   roi_feature shape = [batch_i, max_roi_num, feature_dims]
    #   roi_mask shape = [batch_i, max_roi_num]

    roi_mask = tf.sequence_mask(
        roi_num,
        maxlen=utils.get_tensor_shape(roi_feature)[1],
        dtype=tf.float32)
    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      image_repr = tf.contrib.layers.fully_connected(
          inputs=roi_feature,
          num_outputs=options.embedding_dims,
          activation_fn=None,
          normalizer_fn=lambda x: x)

    tf.summary.histogram('gcn/roi_num', roi_num)

    # Slogan representation.
    #   slogan_num shape = [batch_i]
    #   slogan_text_feature shape = [batch_i, max_slogan_num, max_slogan_length, embedding_dims]
    #   slogan_text_length shape = [batch_i, max_slogan_num]
    #   slogan_repr shape = [batch_i, max_slogan_num, embedding_dims]
    #   slogan_mask shape = [batch_i, max_slogan_num]

    slogan_length_mask = tf.sequence_mask(
        slogan_text_length,
        maxlen=utils.get_tensor_shape(slogan_text_feature)[2],
        dtype=tf.float32)
    slogan_repr = tf.squeeze(
        utils.masked_avg_nd(
            data=slogan_text_feature, mask=slogan_length_mask, dim=2),
        axis=[2])
    slogan_mask = tf.sequence_mask(
        slogan_num,
        maxlen=utils.get_tensor_shape(slogan_text_feature)[1],
        dtype=tf.float32)

    # Slogan-kb representation.

    slogan_kb_length_mask = tf.sequence_mask(
        slogan_kb_text_length,
        maxlen=utils.get_tensor_shape(slogan_kb_text_feature)[2],
        dtype=tf.float32)
    slogan_kb_repr = tf.squeeze(
        utils.masked_avg_nd(
            data=slogan_kb_text_feature, mask=slogan_kb_length_mask, dim=2),
        axis=[2])
    slogan_kb_mask = tf.sequence_mask(
        slogan_kb_num,
        maxlen=utils.get_tensor_shape(slogan_kb_text_feature)[1],
        dtype=tf.float32)

    # Build the message-passing graph.

    nodes = self._node_embedding(
        feature_list=[image_repr, slogan_repr, slogan_kb_repr],
        batch_norm=True,
        is_training=is_training)

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      adjacency, adjacency_logits = self._adjacency_matrix(
          feature1=roi_feature,
          mask1=roi_mask,
          feature2=slogan_repr,
          mask2=slogan_mask,
          feature3=slogan_kb_repr,
          mask3=slogan_kb_mask,
          inference_dims=options.inference_dims,
          connect_roi_to_slogan=options.connect_roi_to_slogan,
          scope='adjacency')

      # GCN layer 1.

      nodes = tf.matmul(adjacency, nodes)

      # # GCN layer 2.

      # raw_nodes = nodes
      # nodes = slim.dropout(tf.nn.relu6(nodes), 0.5, is_training=is_training)
      # nodes = tf.contrib.layers.fully_connected(
      #     inputs=nodes,
      #     num_outputs=nodes.get_shape()[-1].value,
      #     activation_fn=None,
      #     scope='gcn_layer2')
      # nodes = tf.matmul(adjacency, nodes)

      # nodes = raw_nodes + nodes # residual learning

      # GCN layer 2.

      raw_branch = nodes
      nodes = tf.contrib.layers.fully_connected(
          inputs=nodes,
          num_outputs=options.embedding_dims,
          activation_fn=tf.nn.relu6,
          scope='res_a')
      nodes = slim.dropout(
          nodes, options.dropout_keep_prob, is_training=is_training)
      nodes = tf.contrib.layers.fully_connected(
          inputs=nodes,
          num_outputs=options.embedding_dims,
          activation_fn=None,
          scope='res_b')
      nodes = tf.matmul(adjacency, nodes)

      nodes = raw_branch + nodes  # residual learning

      output_mask = tf.concat([roi_mask, tf.zeros_like(slogan_mask), tf.zeros_like(slogan_kb_mask)], axis=1)
      #output_mask = tf.concat([roi_mask, slogan_mask], axis=1)
      graph_repr = utils.masked_avg_nd(data=nodes, mask=output_mask, dim=1)
      graph_repr = tf.squeeze(graph_repr, axis=[1])

    image_l2 = tf.nn.l2_normalize(graph_repr, axis=-1)

    tf.summary.histogram('gcn/slogan_num', slogan_num)
    tf.summary.histogram('gcn/graph_repr', graph_repr)

    # Statement representation.
    #   stmt_text_feature shape = [batch_c, max_stmt_length, embedding_dims]
    #   stmt_mask shape = [batch_c, max_stmt_length]

    stmt_mask = tf.sequence_mask(
        stmt_text_length,
        maxlen=utils.get_tensor_shape(stmt_text_feature)[1],
        dtype=tf.float32)

    stmt_repr = tf.squeeze(
        utils.masked_avg_nd(data=stmt_text_feature, mask=stmt_mask, dim=1),
        axis=[1])
    tf.summary.histogram('gcn/stmt_repr', stmt_repr)
    stmt_l2 = tf.nn.l2_normalize(stmt_repr, axis=-1)

    # Similarity computation.

    dot_product = tf.multiply(
        tf.expand_dims(image_l2, axis=1), tf.expand_dims(stmt_l2, axis=0))
    dot_product = tf.contrib.layers.dropout(
        dot_product,
        keep_prob=options.dropout_keep_prob,
        is_training=is_training)
    similarity = tf.reduce_sum(dot_product, axis=-1)

    predictions.update({
        _FIELD_IMAGE_ID: image_id,
        _FIELD_IMAGE_IDS_GATHERED: image_ids_gathered,
        _FIELD_SIMILARITY: similarity,
        _FIELD_ADJACENCY: adjacency,
        _FIELD_ADJACENCY_LOGITS: adjacency_logits,
    })
    return predictions

  def build_loss(self, predictions, **kwargs):
    """Build tf graph to compute loss.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      loss_dict: dict of loss tensors keyed by name.
    """
    options = self._model_proto

    (image_id, image_ids_gathered,
     similarity) = (predictions[_FIELD_IMAGE_ID],
                    predictions[_FIELD_IMAGE_IDS_GATHERED],
                    predictions[_FIELD_SIMILARITY])

    distance = 1.0 - similarity

    pos_mask = tf.cast(
        tf.equal(
            tf.expand_dims(image_id, axis=1),
            tf.expand_dims(image_ids_gathered, axis=0)), tf.float32)
    neg_mask = 1.0 - pos_mask

    if options.triplet_ap_use_avg:
      distance_ap = utils.masked_avg(distance, pos_mask)
    else:
      distance_ap = utils.masked_maximum(distance, pos_mask)

    # negatives_outside: smallest D_an where D_an > D_ap.

    mask = tf.cast(tf.greater(distance, distance_ap), tf.float32)
    mask = mask * neg_mask
    negatives_outside = utils.masked_minimum(distance, mask)

    # negatives_inside: largest D_an.

    negatives_inside = utils.masked_maximum(distance, neg_mask)

    # distance_an: the semihard negatives.

    mask_condition = tf.greater(tf.reduce_sum(mask, axis=1, keepdims=True), 0.0)

    distance_an = tf.where(mask_condition, negatives_outside, negatives_inside)

    # Triplet loss.

    losses = tf.maximum(distance_ap - distance_an + options.triplet_margin, 0)

    tf.summary.scalar('loss/num_loss_examples',
                      tf.count_nonzero(losses, dtype=tf.float32))
    return {'triplet_loss': tf.reduce_mean(losses)}

  def build_evaluation(self, predictions, **kwargs):
    """Build tf graph to evaluate the model.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      eval_metric_ops: dict of metric results keyed by name. The values are the
        results of calling a metric function, namely a (metric_tensor, 
        update_op) tuple. see tf.metrics for details.
    """
    (image_id, image_ids_gathered,
     similarity) = (predictions[_FIELD_IMAGE_ID],
                    predictions[_FIELD_IMAGE_IDS_GATHERED],
                    predictions[_FIELD_SIMILARITY])

    retrieved_index = tf.argmax(similarity, axis=1)
    predicted_alignment = tf.gather(image_ids_gathered,
                                    tf.argmax(similarity, axis=1))

    accuracy, update_op = tf.metrics.accuracy(image_id, predicted_alignment)

    return {'accuracy': (accuracy, update_op)}


register_model_class(advise_gcn_model_v4_pb2.AdViSEGCNV4Model.ext, Model)
