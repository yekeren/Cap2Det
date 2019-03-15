from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from models.model_base import ModelBase
from protos import op_model_pb2

from nets import nets_factory
from nets import vgg
from core import imgproc
from core import utils
from core import plotlib
from core.standard_fields_mz import InputDataFields
from core.standard_fields_mz import OPPredictions
from core.standard_fields_mz import DetectionResultFields
from core.training_utils import build_hyperparams
from core import init_grid_anchors
from models import utils_mz as model_utils
from core import box_utils
from core import builder as function_builder
from core import sequence_encoding

from object_detection.builders import hyperparams_builder

slim = tf.contrib.slim
_EPSILON = 1e-8


class Model(ModelBase):
  """Object Prediction model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of op_model.OPModel
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, op_model_pb2.OPModel):
      raise ValueError('The model_proto has to be an instance of OPModel.')

    options = model_proto

    self._vocabulary_list = model_utils.read_vocabulary(options.vocabulary_file)

    self._open_vocabulary_list = model_utils.read_vocabulary(
        options.open_vocabulary_file)

    with open(options.open_vocabulary_glove_file, 'rb') as fid:
      self._open_vocabulary_initial_embedding = np.load(fid)

    self._num_classes = len(self._vocabulary_list)

    self._text_encoding_fn = sequence_encoding.get_encode_fn(
        options.text_encoding)

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
      indicator = tf.feature_column.input_layer({'name_to_id': class_texts},
                                                feature_columns=[indicator_col])
      labels = tf.cast(indicator[:, :-1] > 0, tf.float32)

      # if isinstance(batch, int):
      #   labels.set_shape([batch, len(vocabulary_list)])
      # else:
      #   labels.set_shape([None, len(vocabulary_list)])

      labels.set_shape([batch, len(vocabulary_list)])

    return labels

  def _encode_tokens(self,
                     tokens,
                     embedding_dims,
                     vocabulary_list,
                     initial_embedding=None,
                     trainable=True,
                     init_width=0.03):
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
      unk_emb = init_width * (np.random.rand(1, embedding_dims) * 2 - 1)
      initial_value = np.concatenate([initial_embedding, unk_emb], axis=0)
    else:
      initial_value = init_width * (
          np.random.rand(1 + len(vocabulary_list), embedding_dims) * 2 - 1)

    embedding_weights = tf.get_variable(
        name='weights',
        initializer=initial_value.astype(np.float32),
        trainable=trainable)
    token_embedding = tf.nn.embedding_lookup(
        embedding_weights, token_ids, max_norm=None)
    tf.summary.histogram('token_embedding', token_embedding)
    return token_ids, token_embedding

  def _extract_text_feature(self,
                            text_strings,
                            text_lengths,
                            vocabulary_list,
                            initial_embedding=None,
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
    token_ids_flatterned, text_features_flattened = self._encode_tokens(
        text_strings_flattented, embedding_dims, vocabulary_list,
        initial_embedding, trainable)

    token_ids = tf.reshape(token_ids_flatterned, [batch, max_text_length])
    text_feature = tf.reshape(text_features_flattened,
                              [batch, max_text_length, embedding_dims])
    return token_ids, text_feature

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

    _, caption_features = self._extract_text_feature(
        examples[InputDataFields.concat_caption_string],
        examples[InputDataFields.concat_caption_length],
        vocabulary_list=self._open_vocabulary_list,
        initial_embedding=self._open_vocabulary_initial_embedding,
        embedding_dims=options.embedding_dims,
        trainable=options.train_word_embedding)

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      caption_text_features = slim.fully_connected(
          caption_features,
          num_outputs=options.projected_dims,
          activation_fn=None)

    caption_text_features = self._text_encoding_fn(
        caption_features,
        examples[InputDataFields.concat_caption_length],
        is_training=is_training)

    with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
      caption_predicting_logits = slim.fully_connected(
          caption_text_features,
          num_outputs=self._num_classes,
          activation_fn=None,
          scope='caption_predicting_logits')

    predictions = {
        OPPredictions.caption_predicting_logits:
        caption_predicting_logits,
        OPPredictions.caption_predicting_labels:
        tf.round(tf.nn.sigmoid(caption_predicting_logits)),
    }

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

    return self._build_prediction(examples)

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
      vocabulary_list = self._vocabulary_list
      mapping = {
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
      }
      vocabulary_list = [mapping.get(cls, cls) for cls in vocabulary_list]

      (image_id, num_captions, caption_strings, caption_lengths,
       object_texts) = (examples[InputDataFields.image_id],
                        examples[InputDataFields.num_captions],
                        examples[InputDataFields.caption_strings],
                        examples[InputDataFields.caption_lengths],
                        examples[InputDataFields.object_texts])
      image_id = tf.string_to_number(image_id, out_type=tf.int64)

      labels_gt = self._extract_class_label(
          class_texts=slim.flatten(object_texts),
          vocabulary_list=vocabulary_list)

      pred_logits = predictions[OPPredictions.caption_predicting_logits]

      tf.summary.histogram('object_prediction/logits', pred_logits)

      # Loss for caption-predicting network.
      label_prediction_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels_gt, logits=pred_logits)

      loss_dict['label_prediction_loss'] = tf.reduce_mean(label_prediction_loss)

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
    options = self._model_proto

    vocabulary_list = self._vocabulary_list
    mapping = {
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
    }
    vocabulary_list = [mapping.get(cls, cls) for cls in vocabulary_list]

    labels = self._extract_class_label(
        class_texts=slim.flatten(examples[InputDataFields.object_texts]),
        vocabulary_list=vocabulary_list)
    preds = predictions[OPPredictions.caption_predicting_labels]
    scores = tf.nn.sigmoid(predictions[OPPredictions.caption_predicting_logits])

    metrics = {}

    metrics["overall_eval/accuracy"] = tf.metrics.accuracy(
        labels=labels, predictions=preds)
    metrics["overall_eval/f1_score"] = tf.contrib.metrics.f1_score(
        labels=labels, predictions=scores)
    metrics["overall_eval/precision"] = tf.metrics.precision(
        labels=labels, predictions=preds)
    metrics["overall_eval/recall"] = tf.metrics.recall(
        labels=labels, predictions=preds)

    labels_per_class = tf.unstack(labels, axis=1)
    scores_per_class = tf.unstack(scores, axis=1)
    preds_per_class = tf.unstack(preds, axis=1)

    thresholds = (0.1, 0.3, 0.5, 0.7, 0.9)
    precision_at_thresholds = tf.metrics.precision_at_thresholds(
        labels=labels, predictions=scores, thresholds=thresholds)
    metrics["eval/precision_at_thresholds"] = precision_at_thresholds

    for i, t in enumerate(thresholds):
      tf.summary.scalar("eval/precision_at_threshold_%s" % t,
                        precision_at_thresholds[0][i])

    for (c, l, p, s) in zip(vocabulary_list, labels_per_class, preds_per_class,
                            scores_per_class):
      metrics["per_class_accuracy/%s" % c] = tf.metrics.accuracy(
          labels=l, predictions=p)
      metrics["per_class_f1_score/%s" % c] = tf.contrib.metrics.f1_score(
          labels=l, predictions=s)
      metrics["per_class_precision/%s" % c] = tf.metrics.precision(
          labels=l, predictions=p)
      metrics["per_class_recall/%s" % c] = tf.metrics.recall(
          labels=l, predictions=p)

    return metrics
