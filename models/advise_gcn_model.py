from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf

from models.model_base import ModelBase
from protos import advise_gcn_model_pb2

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

_INIT_WIDTH = 0.03


class Model(ModelBase):
  """ADVISE model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of advise_gcn_model_pb2.AdViSEGCN
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, advise_gcn_model_pb2.AdViSEGCNModel):
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
                                ('slgn_word_embedding', self._slgn_vocab_list)]:
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

  def _calc_mlp(self,
                feature,
                mask,
                aggregate=True,
                hidden_layers=0,
                hidden_units=200,
                output_layer=True,
                output_units=200,
                output_activation_fn=None,
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
              inputs=net, num_outputs=hidden_units)
          net = slim.dropout(net, dropout_keep_prob, is_training=is_training)

      if output_layer:
        with tf.variable_scope('output'):
          net = tf.contrib.layers.fully_connected(
              inputs=net,
              num_outputs=output_units,
              activation_fn=output_activation_fn)

    if not aggregate:
      return net

    output = utils.masked_avg_nd(data=net, mask=mask, dim=1)
    return tf.squeeze(output, axis=[1])

  def _pairwise_transform(self, feature1, feature2, dims=100, scope=None):
    """Builds attention model to compute logits.

    Args:
      feature1: A [batch, max_node_num1, dims1] float tensor.
      feature2: A [batch, max_node_num2, dims2] float tensor.

    Returns:
      feature1_to_feature2 shape = [batch, max_node_num1, max_node_num1]
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

    # with tf.variable_scope(scope):
    #   weights = tf.get_variable(
    #       name='weights'.format(scope),
    #       shape=[dims1, dims2],
    #       trainable=True,
    #       initializer=tf.initializers.random_normal(mean=0.0, stddev=0.003))
    #   #weights = weights + 0.01 * tf.eye(num_rows=dims1, num_columns=dims2)

    # feature1_reshaped = tf.reshape(feature1, [-1, dims1])
    # dot_product = tf.matmul(feature1_reshaped, weights)
    # dot_product = tf.reshape(dot_product, [batch1, max_node_num1, dims2])

    # dot_product = tf.matmul(dot_product, feature2, transpose_b=True)
    # dot_product = tf.div(dot_product, np.sqrt(dims1))
    # return dot_product

    # dims = feature2.get_shape()[-1].value
    # feature1_proj = tf.contrib.layers.fully_connected(
    #     inputs=feature1, num_outputs=dims, activation_fn=None, scope=scope)
    # dot_product = tf.matmul(feature1_proj, feature2, transpose_b=True)
    # dot_product = tf.div(dot_product, np.sqrt(dims))

    # return dot_product

  def _calc_graph_representation(self,
                                 feature0,
                                 feature1,
                                 mask1,
                                 feature2,
                                 mask2,
                                 hidden_layers=0,
                                 hidden_units=200,
                                 output_units=200,
                                 dropout_keep_prob=0.5,
                                 is_training=False,
                                 scope=None):
    """Builds representation using Graph Convolutional Network.

    Assume that the messages are passing from feature2 to feature 1.
    Assume that the final representation is the averaging of the feature 1.

    Args:
      feature1: A [batch, max_node_num1, dims] float tensor.
      mask1: A [batch, max_node_num1] boolean tensor.
      feature2: A [batch, max_node_num2, dims] float tensor.
      mask2: A [batch, max_node_num2] boolean tensor.

    Returns:
      A [batch, dims] tensor.
    """
    _, max_node_num1, dims1 = utils.get_tensor_shape(feature1)
    _, max_node_num2, dims2 = utils.get_tensor_shape(feature2)

    assert dims1 == dims2
    dims = dims1

    # Concatenate: get the nodes' representation/mask.
    #   feature shape = [batch, max_node_num1 + max_node_num2, dims]
    #   mask shape = [batch, max_node_num1 + max_node_num2]

    feature1 = tf.nn.l2_normalize(feature1, axis=-1)
    feature2 = tf.nn.l2_normalize(feature2, axis=-1)
    feature = tf.concat([feature1, feature2], axis=1)
    mask = tf.concat([mask1, mask2], axis=1)

    # Compute the adjacency matrix, including two parts:
    #   adjacency1 (feature1 self-connectivity) shape = [batch, max_node_num1, max_node_num1]
    #   adjacency2 (feature1-feature2 connectivity) shape = [batch, max_node_num1, max_node_num2]

    # f1 x W, shape = [batch, max_node_num1, dims]

    with tf.variable_scope(scope):
      feature1_to_feature1 = self._pairwise_transform(
          feature0, feature0, scope='to_feature1')
      feature1_to_feature2 = self._pairwise_transform(
          feature0, feature2, scope='to_feature2')

    tf.summary.histogram('gcn/feature1_to_feature1', feature1_to_feature1)
    tf.summary.histogram('gcn/feature1_to_feature2', feature1_to_feature2)

    # Compute adjacency matrix
    #   shape = [batch, max_node_num1, max_node_num1 + max_node_num2]

    adjacency_logits = tf.concat([feature1_to_feature1, feature1_to_feature2],
                                 axis=2)
    adjacency = utils.masked_softmax(
        data=adjacency_logits, mask=tf.expand_dims(mask, 1), dim=2)
    tf.summary.histogram('gcn/adjacency_logits', adjacency_logits)
    tf.summary.histogram('gcn/adjacency_probas', adjacency)
    adjacency = tf.multiply(adjacency, tf.expand_dims(mask1, axis=2))

    # Pad adjacency matrix
    #   shape = [batch, max_node_num1 + max_node_num2, max_node_num1 + max_node_num2]

    adjacency = tf.pad(
        adjacency, [[0, 0], [0, max_node_num2], [0, 0]],
        mode='CONSTANT',
        constant_values=0)
    _, row, col = utils.get_tensor_shape(adjacency)
    assert_op = tf.Assert(
        tf.logical_and(
            tf.equal(row, max_node_num1 + max_node_num2),
            tf.equal(col, max_node_num1 + max_node_num2)),
        [tf.shape(adjacency), max_node_num1, max_node_num2])

    # Graph convolutional neural network.
    #   adjacency shape = [batch, max_node_num1 + max_node_num2, max_node_num1 + max_node_num2]
    #   feature shape = [batch, max_node_num1 + max_node_num2, dims]
    #   mask shape = [batch, max_node_num1 + max_node_num2]

    net = feature

    with tf.control_dependencies([assert_op]):
      # Layer 1.

      net = tf.matmul(adjacency, net)
      tf.summary.histogram('gcn/layer1', net)

      # # Layer 2.

      # net = slim.batch_norm(
      #     net, activation_fn=tf.nn.relu, is_training=is_training)
      # net = slim.dropout(net, dropout_keep_prob, is_training=is_training)

      # net = tf.contrib.layers.fully_connected(
      #     inputs=net,
      #     num_outputs=net.get_shape()[-1].value,
      #     activation_fn=None,
      #     normalizer_fn=lambda x: x)
      # net = tf.matmul(adjacency, net)
      # tf.summary.histogram('gcn/layer2', net)

    # Average the roi feature.

    avg_mask = tf.concat([mask1, tf.zeros_like(mask2)], axis=1)
    output = utils.masked_avg_nd(data=net, mask=avg_mask, dim=1)
    return tf.squeeze(output, axis=[1]), adjacency, adjacency_logits, mask

    ## Use the image feature.

    #return net[:, 0, :], adjacency, adjacency_logits, mask

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
     roi_feature_size) = (examples[InputDataFields.image_id],
                          examples[InputDataFields.image_feature],
                          examples[InputDataFields.roi_feature],
                          examples[InputDataFields.roi_feature_size])
    (slogan_text_size, slogan_text_string,
     slogan_text_length) = (examples[InputDataFields.slogan_text_size],
                            examples[InputDataFields.slogan_text_string],
                            examples[InputDataFields.slogan_text_length])
    (groundtruth_text_size, groundtruth_text_string,
     groundtruth_text_length) = (
         examples[InputDataFields.groundtruth_text_size],
         examples[InputDataFields.groundtruth_text_string],
         examples[InputDataFields.groundtruth_text_length])

    if options.add_image_as_a_roi:
      roi_feature_size = tf.add(roi_feature_size, 1)
      roi_feature = tf.concat(
          [tf.expand_dims(image_feature, axis=1), roi_feature], axis=1)

    # Generate in-batch question list.

    if is_training:

      # Sample in-batch negatives for the training mode.

      (image_ids_gathered, stmt_text_string,
       stmt_text_length) = model_utils.gather_in_batch_captions(
           image_id, groundtruth_text_size, groundtruth_text_string,
           groundtruth_text_length)
    else:

      # Use the negatives in the tf.train.Example for the evaluation mode.

      batch = image_id.get_shape()[0].value
      assert batch == 1, "We only support `batch == 1` for evaluation."

      (question_text_size, question_text_string,
       question_text_length) = (examples[InputDataFields.question_text_size],
                                examples[InputDataFields.question_text_string],
                                examples[InputDataFields.question_text_length])

      stmt_text_string = question_text_string[0][:question_text_size[0], :]
      stmt_text_length = question_text_length[0][:question_text_size[0]]

      groundtruth_mask = self._mask_groundtruth(
          groundtruth_strings=groundtruth_text_string[0]
          [:groundtruth_text_size[0], :],
          question_strings=question_text_string[0][:question_text_size[0], :])

      image_ids_gathered = tf.where(
          groundtruth_mask,
          x=tf.fill(tf.shape(groundtruth_mask), image_id[0]),
          y=tf.fill(
              tf.shape(groundtruth_mask), tf.constant(-1, dtype=tf.int64)))

    # Word embedding process and inference embedding process.

    max_norm = None
    if options.HasField('max_norm'):
      max_norm = options.max_norm

    stmt_text_feature = tf.nn.embedding_lookup(
        params=self._word_embedding(
            self._stmt_vocab_list,
            options.embedding_dims,
            init_width=_INIT_WIDTH,
            scope='stmt_word_embedding'),
        ids=tf.contrib.lookup.index_table_from_tensor(
            self._stmt_vocab_list, num_oov_buckets=1).lookup(stmt_text_string),
        max_norm=max_norm)

    slogan_text_feature = tf.nn.embedding_lookup(
        params=self._word_embedding(
            self._slgn_vocab_list,
            options.embedding_dims,
            init_width=_INIT_WIDTH,
            scope='slgn_word_embedding'),
        ids=tf.contrib.lookup.index_table_from_tensor(
            self._slgn_vocab_list,
            num_oov_buckets=1).lookup(slogan_text_string),
        max_norm=max_norm)

    # Image representation.
    #   roi_feature_size shape = [batch_i]
    #   roi_feature shape = [batch_i, max_roi_size, feature_dims]
    #   roi_mask shape = [batch_i, max_roi_size]

    roi_mask = tf.sequence_mask(
        roi_feature_size,
        maxlen=utils.get_tensor_shape(roi_feature)[1],
        dtype=tf.float32)
    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      image_repr = self._calc_mlp(
          feature=roi_feature,
          mask=roi_mask,
          aggregate=False,
          hidden_layers=0,
          output_layer=True,
          output_units=options.output_units,
          is_training=is_training,
          scope='image')

    tf.summary.histogram('gcn/input_image_repr', image_repr)
    tf.summary.histogram('gcn/input_slogan_text_feature', slogan_text_feature)
    tf.summary.histogram('gcn/input_stmt_text_feature', stmt_text_feature)

    # Slogan representation.
    #   slogan_text_size shape = [batch_i]
    #   slogan_text_feature shape = [batch_i, max_slogan_size, max_slogan_length, embedding_dims]
    #   slogan_text_length shape = [batch_i, max_slogan_size]
    #   slogan_mask1 shape = [batch_i, max_slogan_size, max_slogan_length]
    #   slogan_repr1 shape = [batch_i, max_slogan_size, embedding_dims]
    #   slogan_repr2 shape = [batch_i, max_slogan_size]
    #   slogan_mask2 shape = [batch_i, embedding_dims]

    slogan_mask1 = tf.sequence_mask(
        slogan_text_length,
        maxlen=utils.get_tensor_shape(slogan_text_feature)[2],
        dtype=tf.float32)
    slogan_repr1 = tf.squeeze(
        utils.masked_avg_nd(data=slogan_text_feature, mask=slogan_mask1, dim=2),
        axis=[2])
    slogan_mask2 = tf.sequence_mask(
        slogan_text_size,
        maxlen=utils.get_tensor_shape(slogan_text_feature)[1],
        dtype=tf.float32)
    slogan_repr2 = tf.squeeze(
        utils.masked_avg_nd(data=slogan_repr1, mask=slogan_mask2, dim=1),
        axis=[1])
    slogan_repr = slogan_repr1

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      graph_repr, adjacency, adjacency_logits, mask = self._calc_graph_representation(
          feature0=roi_feature,
          feature1=image_repr,
          mask1=roi_mask,
          feature2=slogan_repr,
          mask2=slogan_mask2,
          scope='graph')

    tf.summary.histogram('gcn/slogan_size', slogan_text_size)
    tf.summary.histogram('gcn/image_repr', image_repr)
    tf.summary.histogram('gcn/slogan_repr', slogan_repr)
    tf.summary.histogram('gcn/graph_repr', graph_repr)
    image_l2 = tf.nn.l2_normalize(graph_repr, axis=-1)

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
        keep_prob=options.dot_product_dropout_keep_prob,
        is_training=is_training)
    similarity = tf.reduce_sum(dot_product, axis=-1)

    predictions.update({
        'adjacency': adjacency,
        'adjacency_logits': adjacency_logits,
        'mask': mask,
        'roi_mask': roi_mask,
        'slogan_mask': slogan_mask2,
        _FIELD_IMAGE_ID: image_id,
        _FIELD_IMAGE_IDS_GATHERED: image_ids_gathered,
        _FIELD_SIMILARITY: similarity
    })
    # if adjacency is not None:
    #   predictions.update({
    #       'adjacency': adjacency,
    #       'feature1': feature1,
    #       'feature2': feature2,
    #   })
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

    #distance_ap = utils.masked_avg(distance, pos_mask)
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


register_model_class(advise_gcn_model_pb2.AdViSEGCNModel.ext, Model)
