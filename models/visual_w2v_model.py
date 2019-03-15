from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from models.model_base import ModelBase
from protos import visual_w2v_model_pb2

from nets import nets_factory
from nets import vgg
from core import imgproc
from core import utils
from core import plotlib
from core.standard_fields import InputDataFields
from core.standard_fields import VisualW2vPredictions
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
      model_proto: an instance of visual_w2v_model_pb2.VisualW2vModel
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, visual_w2v_model_pb2.VisualW2vModel):
      raise ValueError(
          'The model_proto has to be an instance of VisualW2vModel.')

    options = model_proto

    self._open_vocabulary_list = model_utils.read_vocabulary(
        options.open_vocabulary_file)
    with open(options.open_vocabulary_glove_file, 'rb') as fid:
      self._open_vocabulary_initial_embedding = np.load(fid)

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

    # Image CNN features.

    inputs = examples[InputDataFields.image]
    image_features = model_utils.calc_cnn_feature(
        inputs, options.cnn_options, is_training=is_training)

    with slim.arg_scope(
        build_hyperparams(options.image_fc_hyperparams, is_training)):
      image_features = slim.fully_connected(
          image_features,
          num_outputs=options.shared_dims,
          activation_fn=None,
          scope='image')

    # Text Global-Average-Pooling features.

    (image_id, num_captions, caption_strings,
     caption_lengths) = (examples[InputDataFields.image_id],
                         examples[InputDataFields.num_captions],
                         examples[InputDataFields.caption_strings],
                         examples[InputDataFields.caption_lengths])
    image_id = tf.string_to_number(image_id, out_type=tf.int64)

    (image_ids_gathered, caption_strings_gathered,
     caption_lengths_gathered) = model_utils.gather_in_batch_captions(
         image_id, num_captions, caption_strings, caption_lengths)

    (caption_token_ids_gathered,
     caption_features_gathered) = self._extract_text_feature(
         caption_strings_gathered,
         caption_lengths_gathered,
         vocabulary_list=self._open_vocabulary_list,
         initial_embedding=self._open_vocabulary_initial_embedding,
         embedding_dims=options.embedding_dims,
         trainable=options.train_word_embedding,
         max_norm=None)

    with slim.arg_scope(
        build_hyperparams(options.text_fc_hyperparams, is_training)):
      if visual_w2v_model_pb2.VisualW2vModel.ATT == options.text_feature_extractor:
        attn = slim.fully_connected(
            caption_features_gathered,
            num_outputs=1,
            activation_fn=None,
            scope='caption_attn')
        attn = tf.squeeze(attn, axis=-1)
      caption_features_gathered = slim.fully_connected(
          caption_features_gathered,
          num_outputs=options.shared_dims,
          activation_fn=None,
          scope='caption')

    oov = len(self._open_vocabulary_list)
    caption_masks_gathered = tf.logical_not(
        tf.equal(caption_token_ids_gathered, oov))
    caption_masks_gathered = tf.to_float(caption_masks_gathered)

    if visual_w2v_model_pb2.VisualW2vModel.GAP == options.text_feature_extractor:
      caption_features_gathered = utils.masked_avg_nd(
          data=caption_features_gathered, mask=caption_masks_gathered, dim=1)
      caption_features_gathered = tf.squeeze(caption_features_gathered, axis=1)
    elif visual_w2v_model_pb2.VisualW2vModel.ATT == options.text_feature_extractor:
      attn = utils.masked_softmax(attn, mask=caption_masks_gathered, dim=-1)
      caption_features_gathered = tf.multiply(
          tf.expand_dims(attn, axis=-1), caption_features_gathered)
      caption_features_gathered = utils.masked_sum_nd(
          caption_features_gathered, mask=caption_masks_gathered, dim=1)
      caption_features_gathered = tf.squeeze(caption_features_gathered, axis=1)
    else:
      raise ValueError('Invalid text feature extractor.')

    # Export token embeddings.

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      _, token_embeddings = self._encode_tokens(
          tokens=tf.constant(self._open_vocabulary_list),
          embedding_dims=options.embedding_dims,
          vocabulary_list=self._open_vocabulary_list,
          initial_embedding=self._open_vocabulary_initial_embedding,
          trainable=options.train_word_embedding)
      with slim.arg_scope(
          build_hyperparams(options.text_fc_hyperparams, is_training)):
        token_embeddings = slim.fully_connected(
            token_embeddings,
            num_outputs=options.shared_dims,
            activation_fn=None,
            scope='caption')
    var_to_assign = tf.get_variable(
        name='weights_proj',
        shape=[len(self._open_vocabulary_list), options.shared_dims])
    var_to_assign = tf.assign(var_to_assign, token_embeddings)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, var_to_assign)

    tf.summary.histogram('token_embedding_proj', token_embeddings)

    # Compute similarity.

    similarity = model_utils.calc_pairwise_similarity(
        feature_a=image_features,
        feature_b=caption_features_gathered,
        l2_normalize=True,
        dropout_keep_prob=options.cross_modal_dropout_keep_prob,
        is_training=is_training)

    predictions = {
        VisualW2vPredictions.image_id: image_id,
        VisualW2vPredictions.image_ids_gathered: image_ids_gathered,
        VisualW2vPredictions.similarity: similarity,
        VisualW2vPredictions.word2vec: var_to_assign,
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

    loss_dict = {}

    # Extracts tensors and shapes.

    (image_id, image_ids_gathered,
     similarity) = (predictions[VisualW2vPredictions.image_id],
                    predictions[VisualW2vPredictions.image_ids_gathered],
                    predictions[VisualW2vPredictions.similarity])

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

    # Triplet loss.

    losses = tf.maximum(distance_ap - distance_an + options.triplet_loss_margin,
                        0)

    num_loss_examples = tf.count_nonzero(losses, dtype=tf.float32)
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
    # Extracts tensors and shapes.

    (image_id, image_ids_gathered,
     similarity) = (predictions[VisualW2vPredictions.image_id],
                    predictions[VisualW2vPredictions.image_ids_gathered],
                    predictions[VisualW2vPredictions.similarity])

    # Process retrieval on the in-batch eval dataset.

    retrieved_index = tf.argmax(similarity, axis=1)
    predicted_alignment = tf.gather(image_ids_gathered,
                                    tf.argmax(similarity, axis=1))

    # Calculate accuracy.

    accuracy, update_op = tf.metrics.accuracy(image_id, predicted_alignment)
    return {'accuracy': (accuracy, update_op)}
