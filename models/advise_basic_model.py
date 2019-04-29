from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf

from models.model_base import ModelBase
from protos import advise_basic_model_pb2

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


class Model(ModelBase):
  """ADVISE model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of advise_basic_model_pb2.AdViSEGCN
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, advise_basic_model_pb2.AdViSEBasicModel):
      raise ValueError('The model_proto has to be an instance of AdViSEGCN.')

    options = model_proto

    # Read vocabulary.

    vocab_with_freq = model_utils.read_vocabulary_with_frequency(
        options.stmt_vocab_list_path)
    self._vocab_list = [word for word, freq in vocab_with_freq if freq > 10]

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

      word_embedding = [
          word2vec.get(word, 0.01 * (np.random.rand(dims) * 2.0 - 1.0))
          for word in self._vocab_list
      ]
      word_embedding = np.stack(word_embedding, axis=0)

      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        var = tf.get_variable(
            name='word_embedding/weights',
            shape=[len(self._vocab_list), dims],
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

  def _word_embedding(self, vocab_list, embedding_dims=200):
    """Gets the word embedding.

    Args:
      vocab_list: A list of string.

    Returns:
      word_embedding: A tensor of shape [1 + number_of_tokens, dims].
    """
    word_embedding = tf.get_variable(
        name='word_embedding/weights',
        shape=[len(vocab_list), embedding_dims],
        trainable=True)
    oov_embedding = tf.get_variable(
        name='word_embedding/oov_weights',
        initializer=tf.initializers.random_uniform(minval=-0.03, maxval=0.03),
        shape=[1, embedding_dims],
        trainable=True)
    word_embedding = tf.concat([word_embedding, oov_embedding], axis=0)
    return word_embedding

  def _dense_connect(self, mask):
    """Gets the adjacency matrix of a densely connected graph.

    Args:
      mask: A [batch, max_node_num] boolean tensor.

    Returns:
      adjacency: A [batch, max_node_num, max_node_num] float tensor.
    """
    mask1 = tf.expand_dims(mask, axis=1)
    mask2 = tf.expand_dims(mask, axis=2)
    adjacency = mask1 * mask2
    sum_val = utils.masked_sum(data=adjacency, mask=mask1, dim=2)
    adjacency = tf.div(adjacency, tf.maximum(sum_val, _EPSILON))
    return adjacency

  def _similarity_connect(self, feature, mask, project_dims=200):
    """Gets the adjacency matrix of a graph, depends on the nodes similarity.

    Args:
      feature: A [batch, max_node_num, dims] float tensor.
      mask: A [batch, max_node_num] boolean tensor.

    Returns:
      adjacency: A [batch, max_node_num, max_node_num] float tensor.
    """
    tf.summary.histogram('gcn/feature', feature)

    with tf.variable_scope('project_1'):
      feature1 = tf.contrib.layers.fully_connected(
          inputs=feature,
          num_outputs=project_dims,
          activation_fn=None,
          normalizer_fn=lambda x: x)
    with tf.variable_scope('project_2'):
      feature2 = tf.contrib.layers.fully_connected(
          inputs=feature,
          num_outputs=project_dims,
          activation_fn=None,
          normalizer_fn=lambda x: x)
    tf.summary.histogram('gcn/project1', feature1)
    tf.summary.histogram('gcn/project2', feature2)

    dot_product = tf.matmul(feature1, feature2, transpose_b=True)

    mask1 = tf.expand_dims(mask, axis=1)
    mask2 = tf.expand_dims(mask, axis=2)

    adjacency = utils.masked_softmax(data=dot_product, mask=mask1, dim=2)
    adjacency = tf.multiply(adjacency, mask2)

    tf.summary.histogram('gcn/logits', dot_product)
    tf.summary.histogram('gcn/adjacency', adjacency)
    return adjacency

  def _calc_graph_representation(self,
                                 feature,
                                 mask,
                                 adjacency=None,
                                 hidden_layers=0,
                                 hidden_units=200,
                                 output_units=200,
                                 dropout_keep_prob=0.5,
                                 is_training=False,
                                 scope=None):
    """Builds representation using Graph Convolutional Network.

    Args:
      feature: A [batch, max_node_num, dims] float tensor.
      adjacency: A [batch, max_node_num, max_node_num] float tensor.
      mask: A [batch, max_node_num] boolean tensor.

    Returns:
      A [batch, dims] tensor.
    """
    net = feature
    with tf.variable_scope(scope):
      for i in range(hidden_layers):
        with tf.variable_scope('hidden_{}'.format(i)):
          net = tf.contrib.layers.fully_connected(
              inputs=net,
              num_outputs=hidden_units,
              activation_fn=None,
              normalizer_fn=lambda x: x)
          if adjacency is not None:
            net = tf.matmul(adjacency, net)
          net = slim.batch_norm(
              net, activation_fn=tf.nn.relu, is_training=is_training)
          net = slim.dropout(net, dropout_keep_prob, is_training=is_training)

      with tf.variable_scope('output'):
        net = tf.contrib.layers.fully_connected(
            inputs=net,
            num_outputs=output_units,
            activation_fn=None,
            normalizer_fn=lambda x: x)
        if adjacency is not None:
          net = tf.matmul(adjacency, net)
        net = slim.batch_norm(net, activation_fn=None, is_training=is_training)
    output = utils.masked_avg_nd(data=net, mask=mask, dim=1)
    return tf.squeeze(output, axis=[1])

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

    (image_id, image_feature, roi_feature, roi_feature_size,
     groundtruth_text_size, groundtruth_text_string,
     groundtruth_text_length) = (
         examples[InputDataFields.image_id],
         examples[InputDataFields.image_feature],
         examples[InputDataFields.roi_feature],
         examples[InputDataFields.roi_feature_size],
         examples[InputDataFields.groundtruth_text_size],
         examples[InputDataFields.groundtruth_text_string],
         examples[InputDataFields.groundtruth_text_length])

    # Generate in-batch question list.

    if is_training:

      # Sample in-batch negatives for the training mode.

      (image_ids_gathered, groundtruth_text_strings_gathered,
       groundtruth_text_lengths_gathered
      ) = model_utils.gather_in_batch_captions(image_id, groundtruth_text_size,
                                               groundtruth_text_string,
                                               groundtruth_text_length)
    else:

      # Use the negatives in the tf.train.Example for the evaluation mode.

      batch = image_id.get_shape()[0].value
      assert batch == 1, "We only support `batch == 1` for evaluation."

      (question_text_size, question_text_string,
       question_text_length) = (examples[InputDataFields.question_text_size],
                                examples[InputDataFields.question_text_string],
                                examples[InputDataFields.question_text_length])

      groundtruth_text_strings_gathered = question_text_string[
          0][:question_text_size[0], :]
      groundtruth_text_lengths_gathered = question_text_length[
          0][:question_text_size[0]]

      groundtruth_mask = self._mask_groundtruth(
          groundtruth_strings=groundtruth_text_string[0]
          [:groundtruth_text_size[0], :],
          question_strings=question_text_string[0][:question_text_size[0], :])

      image_ids_gathered = tf.where(
          groundtruth_mask,
          x=tf.fill(tf.shape(groundtruth_mask), image_id[0]),
          y=tf.fill(
              tf.shape(groundtruth_mask), tf.constant(-1, dtype=tf.int64)))

    # Add the image feature as a ROI feature.

    if options.add_image_as_a_roi:
      roi_feature_size = tf.add(roi_feature_size, 1)
      roi_feature = tf.concat(
          [tf.expand_dims(image_feature, axis=1), roi_feature], axis=1)

    # Load vocabulary, project words to embeddings.

    word_embedding = self._word_embedding(self._vocab_list,
                                          options.embedding_dims)
    norm = tf.norm(word_embedding, axis=-1)
    tf.summary.scalar('gcn/emb_norm_avg', tf.reduce_mean(norm))
    tf.summary.scalar('gcn/emb_norm_max', tf.reduce_max(norm))
    tf.summary.scalar('gcn/emb_norm_min', tf.reduce_min(norm))

    table = tf.contrib.lookup.index_table_from_tensor(
        self._vocab_list, num_oov_buckets=1)

    max_norm = None
    if options.HasField('max_norm'):
      max_norm = options.max_norm
    groundtruth_text_features_gathered = tf.nn.embedding_lookup(
        word_embedding,
        table.lookup(groundtruth_text_strings_gathered),
        max_norm=max_norm)

    #adjacency = self._dense_connect(mask)
    #adjacency = self._similarity_connect(feature, mask)

    adjacency = None
    roi_mask = tf.sequence_mask(
        roi_feature_size,
        maxlen=utils.get_tensor_shape(roi_feature)[1],
        dtype=tf.float32)
    image_repr = self._calc_graph_representation(
        roi_feature,
        roi_mask,
        adjacency=adjacency,
        hidden_units=options.hidden_units,
        output_units=options.output_units,
        is_training=is_training,
        scope='image')
    image_l2 = tf.nn.l2_normalize(image_repr, axis=-1)

    # Stmt feature aggregation.

    adjacency = None
    stmt_mask = tf.sequence_mask(
        groundtruth_text_lengths_gathered,
        maxlen=utils.get_tensor_shape(groundtruth_text_features_gathered)[1],
        dtype=tf.float32)

    # AVG-pooling.
    stmt_repr = utils.masked_avg_nd(
        data=groundtruth_text_features_gathered, mask=stmt_mask, dim=1)
    stmt_repr = tf.squeeze(stmt_repr, axis=[1])

    #stmt_repr = self._calc_graph_representation(
    #        groundtruth_text_features_gathered,
    #        stmt_mask,
    #        adjacency=adjacency,
    #        hidden_units=options.hidden_units,
    #        output_units=options.output_units,
    #        is_training=is_training,
    #        scope='stmt')
    stmt_l2 = tf.nn.l2_normalize(stmt_repr, axis=-1)

    # Similarity computation.

    dot_product = tf.multiply(
        tf.expand_dims(image_l2, axis=1), tf.expand_dims(stmt_l2, axis=0))
    dot_product = tf.contrib.layers.dropout(
        dot_product,
        keep_prob=options.dot_product_dropout_keep_prob,
        is_training=is_training)
    similarity = tf.reduce_sum(dot_product, axis=-1)

    predictions.update({
        _FIELD_IMAGE_ID: image_id,
        _FIELD_IMAGE_IDS_GATHERED: image_ids_gathered,
        _FIELD_SIMILARITY: similarity
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

    distance_ap = utils.masked_avg(distance, pos_mask)
    #distance_ap = utils.masked_maximum(distance, pos_mask)

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


register_model_class(advise_basic_model_pb2.AdViSEBasicModel.ext, Model)
