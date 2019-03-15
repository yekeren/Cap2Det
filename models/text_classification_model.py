from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from models.model_base import ModelBase
from protos import text_classification_model_pb2

from nets import nets_factory
from nets import vgg
from core import imgproc
from core import utils
from core import plotlib
from core.standard_fields import InputDataFields
from core.standard_fields import TextClassificationPredictions
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
  """VisualW2v model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of text_classification_model_pb2.TextClassificationModel
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto,
                      text_classification_model_pb2.TextClassificationModel):
      raise ValueError(
          'The model_proto has to be an instance of TextClassificationModel.')

    options = model_proto

    self._open_vocabulary_list = model_utils.read_vocabulary(
        options.open_vocabulary_file)
    with open(options.open_vocabulary_glove_file, 'rb') as fid:
      self._open_vocabulary_initial_embedding = np.load(fid)

    self._vocabulary_list = model_utils.read_vocabulary(options.vocabulary_file)

    self._num_classes = len(self._vocabulary_list)

  def _extract_class_label(self, class_texts, vocabulary_list):
    """Extracts class labels.

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

  def _extract_pseudo_label(self,
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

    mapping = {
        # For coco mapping.
        'traffic light': 'stoplight',
        'fire hydrant': 'hydrant',
        'stop sign': 'sign',
        'parking meter': 'meter',
        'sports ball': 'ball',
        'baseball bat': 'bat',
        'baseball glove': 'glove',
        'tennis racket': 'racket',
        'wine glass': 'wineglass',
        'hot dog': 'hotdog',
        'potted plant': 'plant',
        'dining table': 'table',
        'cell phone': 'cellphone',
        'teddy bear': 'teddy',
        'hair drier': 'hairdryer',
        # For pascal mapping.
        'aeroplane': 'airplane',
        'diningtable': 'table',
        'pottedplant': 'plant',
        'tvmonitor': 'tv',
    }

    vocabulary_list = [mapping.get(cls, cls) for cls in vocabulary_list]

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
          trainable=False)

    with tf.variable_scope('token_embedding', reuse=True):
      text_token_ids, text_embeddings = self._extract_text_feature(
          texts,
          None,
          open_vocabulary_list,
          initial_embedding=self._open_vocabulary_initial_embedding,
          embedding_dims=embedding_dims,
          trainable=False)

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
        text_strings_flattented, embedding_dims, vocabulary_list,
        initial_embedding, trainable)

    token_ids = tf.reshape(token_ids_flatterned, [batch, max_text_length])
    text_features = tf.reshape(text_features_flattened,
                               [batch, max_text_length, embedding_dims])
    return token_ids, text_features

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

    # Text Global-Maximum-Pooling features.

    (caption_string,
     caption_length) = (examples[InputDataFields.concat_caption_string],
                        examples[InputDataFields.concat_caption_length])

    (caption_token_ids, caption_features) = self._extract_text_feature(
        caption_string,
        caption_length,
        vocabulary_list=self._open_vocabulary_list,
        initial_embedding=self._open_vocabulary_initial_embedding,
        embedding_dims=options.embedding_dims,
        trainable=options.train_word_embedding,
        max_norm=None)

    with slim.arg_scope(
        build_hyperparams(options.text_fc_hyperparams, is_training)):
      caption_features = slim.fully_connected(
          caption_features,
          num_outputs=self._num_classes,
          activation_fn=None,
          scope='caption')

    oov = len(self._open_vocabulary_list)
    caption_masks = tf.to_float(
        tf.logical_not(tf.equal(caption_token_ids, oov)))

    # logits shape = [batch, num_classes].

    logits = utils.masked_maximum(
        data=caption_features,
        mask=tf.expand_dims(caption_masks, axis=-1),
        dim=1)
    logits = tf.squeeze(logits, axis=1)

    predictions = {
        TextClassificationPredictions.vocab: tf.constant(self._vocabulary_list),
        TextClassificationPredictions.logits: logits,
    }
    return predictions

  def build_loss(self, predictions, examples, **kwargs):
    """Build tf graph to compute loss.

    Args:
      predictions: dict of prediction results keyed by name.
      examples: dict of inputs keyed by name.

    Returns:
      loss_dict: dict of loss tensors keyed by name.
    """
    options = self._model_proto

    # Use inaccurate label to train.

    logits = predictions[TextClassificationPredictions.logits]

    mapping = {
        # For coco mapping.
        'traffic light': 'stoplight',
        'fire hydrant': 'hydrant',
        'stop sign': 'sign',
        'parking meter': 'meter',
        'sports ball': 'ball',
        'baseball bat': 'bat',
        'baseball glove': 'glove',
        'tennis racket': 'racket',
        'wine glass': 'wineglass',
        'hot dog': 'hotdog',
        'potted plant': 'plant',
        'dining table': 'table',
        'cell phone': 'cellphone',
        'teddy bear': 'teddy',
        'hair drier': 'hairdryer',
        # For pascal mapping.
        'aeroplane': 'airplane',
        'diningtable': 'table',
        'pottedplant': 'plant',
        'tvmonitor': 'tv',
    }
    substituted_vocabulary_list = [
        mapping.get(cls, cls) for cls in self._vocabulary_list
    ]

    # Using ground-truth labels.

    if options.label_option == text_classification_model_pb2.TextClassificationModel.GROUNDTRUTH:
      labels = self._extract_class_label(
          class_texts=examples[InputDataFields.object_texts],
          vocabulary_list=self._vocabulary_list)
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)
      loss = tf.reduce_mean(losses)

    # Using exact-matching labels.

    elif options.label_option == text_classification_model_pb2.TextClassificationModel.EXACT_MATCH:
      labels = self._extract_class_label(
          class_texts=slim.flatten(examples[InputDataFields.caption_strings]),
          vocabulary_list=substituted_vocabulary_list)
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)
      loss = tf.reduce_mean(losses)

    elif options.label_option == text_classification_model_pb2.TextClassificationModel.EXPAND_MATCH:
      labels = self._extract_class_label(
          class_texts=slim.flatten(examples[InputDataFields.caption_strings]),
          vocabulary_list=model_utils.expand_vocabulary(self._vocabulary_list))
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)
      loss = tf.reduce_mean(losses)

    # Using exact-matching + w2v-matchiong.

    elif options.label_option == text_classification_model_pb2.TextClassificationModel.EXACT_W2V_MATCH:
      labels_gt = self._extract_class_label(
          class_texts=slim.flatten(examples[InputDataFields.caption_strings]),
          vocabulary_list=substituted_vocabulary_list)
      labels_ps = self._extract_pseudo_label(
          texts=slim.flatten(examples[InputDataFields.caption_strings]),
          vocabulary_list=self._vocabulary_list,
          open_vocabulary_list=self._open_vocabulary_list,
          embedding_dims=options.embedding_dims)
      select_op = tf.reduce_any(labels_gt > 0, axis=-1)
      labels = tf.where(select_op, labels_gt, labels_ps)
      tf.summary.scalar('metrics/num_gt', tf.reduce_sum(tf.to_float(select_op)))

      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)
      loss = tf.reduce_mean(losses)

    # Using w2v-matchiong.

    elif options.label_option == text_classification_model_pb2.TextClassificationModel.W2V_MATCH:
      labels_ps = self._extract_pseudo_label(
          texts=slim.flatten(examples[InputDataFields.caption_strings]),
          vocabulary_list=self._vocabulary_list,
          open_vocabulary_list=self._open_vocabulary_list,
          embedding_dims=options.embedding_dims)

      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels_ps, logits=logits)
      loss = tf.reduce_mean(losses)

    else:
      raise ValueError('not implemented')
    return {'text_cross_entropy_loss': loss}

  def build_evaluation(self, predictions, examples, **kwargs):
    """Build tf graph to evaluate the model.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      eval_metric_ops: dict of metric results keyed by name. The values are the
        results of calling a metric function, namely a (metric_tensor, 
        update_op) tuple. see tf.metrics for details.
    """
    # Use actual label to evaluate.

    logits = predictions[TextClassificationPredictions.logits]
    labels = self._extract_class_label(
        class_texts=examples[InputDataFields.object_texts],
        vocabulary_list=self._vocabulary_list)

    assert labels.get_shape()[0] == 1

    sparse_labels = tf.boolean_mask(
        tf.range(self._num_classes, dtype=tf.int64), labels[0, :] > 0)
    sparse_labels = tf.expand_dims(sparse_labels, axis=0)

    eval_metric_ops = {}

    for k in [1, 5]:
      precision, update_op = tf.metrics.precision_at_k(
          sparse_labels, logits, k=k)
      eval_metric_ops['metrics/precision_at_{}'.format(k)] = (precision,
                                                              update_op)

      recall, update_op = tf.metrics.recall_at_k(sparse_labels, logits, k=k)
      eval_metric_ops['metrics/recall_at_{}'.format(k)] = (recall, update_op)

    return eval_metric_ops
