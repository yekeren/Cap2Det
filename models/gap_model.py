
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from nets import nets_factory

from models.model_base import ModelBase
from protos import gap_model_pb2

from core import utils
from core import plotlib
from core.standard_fields import OperationNames
from core.standard_fields import InputDataFields
from core.standard_fields import GAPVariableScopes
from core.standard_fields import GAPPredictions
from core.standard_fields import GAPPredictionTasks
from models import utils as model_utils

slim = tf.contrib.slim

_BIG_NUMBER = 1e8
_SMALL_NUMBER = 1e-8
_SPLITTER = "-" * 128
_INIT_WIDTH = 0.03
_STDDEV = 0.03


class Model(ModelBase):
  """GAP model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of gap_model_pb2.GAPModel
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, gap_model_pb2.GAPModel):
      raise ValueError('The model_proto has to be an instance of GAPModel.')

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      a list of model variables or None by default.
    """
    options = self._model_proto

    # Filter out CNN variables.
    variables_to_train = []
    for var in tf.trainable_variables():
      if options.cnn_trainable or GAPVariableScopes.cnn not in var.op.name:
        variables_to_train.append(var)
      else:
        tf.logging.info("Freeze cnn parameter %s.", var.op.name)

    # Print trainable variables.

    tf.logging.info(_SPLITTER)
    for var in variables_to_train:
      tf.logging.info("Model variables: %s.", var.op.name)

    return variables_to_train

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
      tf.logging.info(_SPLITTER)
      tf.logging.info("Initialize using scaffold.")

      # Look for the coco_word_embedding.

      variables = tf.trainable_variables(
          'input_layer/' + GAPVariableScopes.word_embedding)
      if len(variables) != 1:
        raise ValueError("The coco_word_embedding should be unique.")
      coco_word_embedding = variables[0]

      # Initialize coco_word_embedding.

      options = self._model_proto
      if options.HasField("vocabulary_weights_file"):
        with open(options.vocabulary_weights_file, "rb") as fp:
          values = np.load(fp)
        dims = values.shape[1]

        oov = _INIT_WIDTH * (np.random.rand(dims) * 2 - 1)
        values = np.concatenate([values, np.expand_dims(oov, 0)], axis=0)

        coco_word_embedding.load(values, sess)
        tf.logging.info("Load embedding weights from %s, shape=%s.", 
            options.vocabulary_weights_file, values.shape)

    scaffold = tf.train.Scaffold(init_fn=_init_fn)
    return scaffold

  def _encode_images(self, 
      image, 
      cnn_name="mobilenet_v2", 
      cnn_trainable=False,
      cnn_weight_decay=1e-4,
      cnn_feature_map="layer_18/output",
      cnn_dropout_keep_prob=1.0,
      cnn_checkpoint=None,
      common_dimensions=300,
      scope="image_proj",
      is_training=False):

    """Builds image model.

    Args:
      image: a [batch, height, width, channels] float tensor, the values are 
        ranging from [0.0, 255.0].
      cnn_name: name of the backbone CNN network.
      cnn_trainable: if True, also train the CNN.
      cnn_weight_decay: weight decay of the CNN networ.
      cnn_feature_map: CNN feature map to be used.
      cnn_dropout_keep_prob: dropout keep probability of the CNN model.
      cnn_checkpoint: path to the pre-trained CNN model.
      common_dimensions: depth of the image embedding.
      scope: variable scope of the projection layer.
      is_training: if True, training graph is built.

    Returns:
      feature_map: a [batch, feature_height, feature_width, common_dimensions] 
        float tensor.
    """
    # Preprocess and extract CNN feature.

    image = image * 2.0 / 255.0 - 1.0
    net_fn = nets_factory.get_network_fn(
        name=cnn_name,
        num_classes=None, 
        weight_decay=cnn_weight_decay, 
        is_training=cnn_trainable and is_training)

    with tf.variable_scope(GAPVariableScopes.cnn):
      _, end_points = net_fn(image)
    feature_map = end_points[cnn_feature_map]

    feature_map = tf.contrib.layers.dropout(
        feature_map, 
        keep_prob=cnn_dropout_keep_prob, 
        is_training=is_training)

    # Add additional projection layer.

    with tf.variable_scope(scope):
      feature_map = tf.contrib.layers.conv2d(
          inputs=feature_map,
          num_outputs=common_dimensions,
          kernel_size=[1, 1],
          activation_fn=None,
          weights_initializer=tf.truncated_normal_initializer(stddev=_STDDEV))

    # Load pre-trained model from checkpoint.

    if cnn_checkpoint is None:
      tf.logging.warning("Pre-trained checkpoint path is not provided!")
    else:
      tf.train.init_from_checkpoint(cnn_checkpoint, 
          assignment_map={"/": GAPVariableScopes.cnn + "/"})

    return feature_map

  def _read_vocabulary(self, filename):
    """Reads vocabulary list from file.

    Args:
      filename: path to the file storing vocabulary info.

    Returns:
      vocabulary_list: a list of string.
    """
    with tf.gfile.GFile(filename, "r") as fid:
      vocabulary_list = [word.strip('\n') for word in fid.readlines()]
    return vocabulary_list

  def _encode_captions(self, 
      caption_strings, 
      vocabulary_list=None,
      common_dimensions=300,
      scope="coco_word_embedding",
      is_training=False):

    """Builds caption model.

    Args:
      caption_strings: captions in the batch, a [num_captions_in_batch,
        max_caption_length] string tensor.
      vocabulary_list: words in the vocabulary, a list of python strings.
      common_dimensions: dimensions of the word embedding.
      is_training: if True, training graph is built.

    Returns:
      text_feature: embedding of each word, a [num_captions_in_batch, 
        max_caption_length, common_dimensions] tensor.
    """

    # Initialize the embedding column.

    if not vocabulary_list:
      raise ValueError('The vocabulary_list cannot be empty.')

    if scope[-len("_embedding"):] != "_embedding":
      raise ValueError("Invalid variable scope name %s.", scope)
    scope_prefix = scope[:-len("_embedding")]

    (categorical_column
     ) = tf.feature_column.categorical_column_with_vocabulary_list(
       key=scope_prefix,
       vocabulary_list=vocabulary_list,
       dtype=tf.string,
       num_oov_buckets=1)
    embedding_column = tf.feature_column.embedding_column(
        categorical_column, dimension=common_dimensions)

    # Embed the caption words.

    (num_captions_in_batch, max_caption_length
     ) = utils.get_tensor_shape(caption_strings)

    caption_strings_flattened = tf.reshape(caption_strings, [-1, 1])

    text_feature_flattened = tf.feature_column.input_layer(
        {scope_prefix: caption_strings_flattened},
        feature_columns=[embedding_column])

    text_feature = tf.reshape(text_feature_flattened, 
        [num_captions_in_batch, max_caption_length, common_dimensions])

    return text_feature

  def _calc_pairwise_similarity(self, 
      image_feature, text_feature, dropout_keep_prob=1.0, is_training=False):

    """Computes the pairwise dot-product similarity between image and caption.

    Args:
      image_feature: image feature, a [batch, num_regions, common_dimensions] 
        float32 tensor.
      text_feature: text feature, a [num_captions_in_batch, max_caption_length, 
        common_dimensions] float32 tensor.
      dropout_keep_prob: dropout keep probability.
      is_training: if True, build the training graph.

    Returns:
      similarity: dot-product similarity, a [batch, num_regions,
        num_captions_in_batch, max_caption_length] float32 tensor.
    """
    if image_feature.get_shape()[-1].value != text_feature.get_shape()[-1].value:
      raise ValueError("The common dimensions of image/text should be matched.")

    image_feature = tf.expand_dims(tf.expand_dims(image_feature, 2), 2)
    text_feature = tf.expand_dims(tf.expand_dims(text_feature, 0), 0)

    dot_product = tf.contrib.layers.dropout(
        image_feature * text_feature,
        keep_prob=dropout_keep_prob,
        is_training=is_training)
    return tf.reduce_sum(dot_product, axis=-1)

  def _calc_saliency_score(self, inputs, 
      is_training=False, use_batch_norm=True, scope="calc_saliency_score"):

    """Calculates saliency score.

    Args:
      inputs: input feature, a [..., feature_dimensions] float tensor.
      is_training: if True, build training graph.
      scope: optional scope for variable scope.

    Returns:
      saliency_score: saliency score, a [..., 1] float tensor keeping the
        feature dimension.
    """
    normalizer_fn = None
    if use_batch_norm:
      normalizer_fn = tf.contrib.layers.batch_norm

    normalizer_params = {
      "decay": 0.995,
      "center": True,
      "scale": True,
      "epsilon": 0.001,
      "is_training": is_training,
    }
    saliency_score = tf.contrib.layers.fully_connected(
        inputs, 
        num_outputs=1, 
        activation_fn=None,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params,
        weights_initializer=tf.truncated_normal_initializer(stddev=_STDDEV),
        scope=scope)
    return tf.squeeze(saliency_score, axis=-1)

  def _predict_image_saliency(self, examples):
    """Builds tf graph for prediction.

    Args:
      examples: dict of input tensors keyed by name.

    Returns:
      predictions: dict of prediction results keyed by name.
    """
    options = self._model_proto
    is_training = self._is_training

    if not options.use_saliency_score:
      raise ValueError("The flag of `use_saliency_score` should be set.")

    image = examples[InputDataFields.image]

    # Extract image feature, shape = 
    #   [batch, feature_height * feature_width, common_dimensions].

    image_feature = self._encode_images(image,
        cnn_name=options.cnn_name,
        cnn_trainable=options.cnn_trainable,
        cnn_weight_decay=options.cnn_weight_decay,
        cnn_feature_map=options.cnn_feature_map,
        cnn_dropout_keep_prob=options.cnn_dropout_keep_prob,
        cnn_checkpoint=options.cnn_checkpoint,
        common_dimensions=options.common_dimensions,
        scope=GAPVariableScopes.image_proj,
        is_training=is_training)

    (batch, feature_height, feature_width, common_dimensions
     ) = utils.get_tensor_shape(image_feature)
    image_feature = tf.reshape(image_feature, [batch, -1, common_dimensions])

    # Predict saliency score.
    #   image_saliency shape = [batch, num_regions].
    #   caption_saliency shape = [num_captions_in_batch, max_caption_length].

    image_saliency = self._calc_saliency_score(
        image_feature, 
        is_training=is_training,
        use_batch_norm=True,
        scope=GAPVariableScopes.image_saliency)
    return { 
      GAPPredictions.image_saliency: tf.reshape(
          image_saliency, [-1, feature_height, feature_width]),
    }

  def _predict_similarity(self, examples):
    """Builds tf graph for prediction.

    Args:
      examples: dict of input tensors keyed by name.

    Returns:
      predictions: dict of prediction results keyed by name.
    """
    options = self._model_proto
    is_training = self._is_training

    (image, image_id, 
     num_captions, caption_strings, caption_lengths) = (
       examples[InputDataFields.image], 
       examples[InputDataFields.image_id],
       examples[InputDataFields.num_captions],
       examples[InputDataFields.caption_strings],
       examples[InputDataFields.caption_lengths])

    (image_ids_gathered, 
     caption_strings_gathered, 
     caption_lengths_gathered) = model_utils.gather_in_batch_captions(
       image_id, num_captions, caption_strings, caption_lengths)

    # Extract image feature, shape = 
    #   [batch, feature_height * feature_width, common_dimensions].

    with tf.name_scope(OperationNames.image_model):
      image_feature = self._encode_images(image,
          cnn_name=options.cnn_name,
          cnn_trainable=options.cnn_trainable,
          cnn_weight_decay=options.cnn_weight_decay,
          cnn_feature_map=options.cnn_feature_map,
          cnn_dropout_keep_prob=options.cnn_dropout_keep_prob,
          cnn_checkpoint=options.cnn_checkpoint,
          common_dimensions=options.common_dimensions,
          scope=GAPVariableScopes.image_proj,
          is_training=is_training)

      (batch, feature_height, feature_width, common_dimensions
       ) = utils.get_tensor_shape(image_feature)
      image_feature = tf.reshape(image_feature, [batch, -1, common_dimensions])

    # Extract caption feature, shape =
    #   [num_captions_in_batch, max_caption_length, common_dimensions].

    vocabulary_list = self._read_vocabulary(options.vocabulary_file)
    tf.logging.info("Read a vocabulary with %i words.", len(vocabulary_list))

    with tf.name_scope(OperationNames.text_model):
      caption_feature = self._encode_captions(
          caption_strings_gathered,
          vocabulary_list=vocabulary_list,
          common_dimensions=options.common_dimensions,
          scope=GAPVariableScopes.word_embedding,
          is_training=is_training)

      (num_captions_in_batch, max_caption_length, common_dimensions
       ) = utils.get_tensor_shape(caption_feature)

    # Calculates similarity matrix, shape=[batch, num_captions_in_batch].

    with tf.name_scope(OperationNames.calc_pairwise_similarity):

      # Compute dot-product similarity.

      similarity = self._calc_pairwise_similarity(
          image_feature=tf.nn.l2_normalize(image_feature, axis=-1),
          text_feature=tf.nn.l2_normalize(caption_feature, axis=-1),
          dropout_keep_prob=options.dropout_keep_prob,
          is_training=is_training)

      word_mask = tf.sequence_mask(
          caption_lengths_gathered, maxlen=max_caption_length, dtype=tf.float32)
      similarity = similarity * tf.expand_dims(tf.expand_dims(word_mask, 0), 0)

      if options.use_saliency_score:
        
        # Predict saliency score.
        #   image_saliency shape = [batch, num_regions].
        #   caption_saliency shape = [num_captions_in_batch, max_caption_length].

        image_saliency = self._calc_saliency_score(
            image_feature, 
            is_training=is_training,
            use_batch_norm=True,
            scope=GAPVariableScopes.image_saliency)
        caption_saliency = self._calc_saliency_score(
            caption_feature,
            is_training=is_training,
            use_batch_norm=True,
            scope=GAPVariableScopes.word_saliency)

        # Apply masked attention.

        image_attention = tf.nn.softmax(image_saliency, axis=-1)
        caption_attention = utils.masked_softmax(
            caption_saliency, word_mask, dim=-1)

        if options.image_regularizer_weight > 0.0:
          log_image_attention = tf.log(
              tf.maximum(image_attention, _SMALL_NUMBER))
          loss = tf.multiply(
              options.image_regularizer_weight,
              tf.reduce_mean(tf.reduce_sum(log_image_attention, axis=1)))
          tf.losses.add_loss(loss)
          tf.summary.scalar('loss/image_attention_log_loss', loss)
          tf.summary.scalar('loss/image_attention_max', 
              tf.reduce_mean(tf.reduce_max(image_attention, axis=1)))
          tf.summary.scalar('loss/image_attention_min', 
              tf.reduce_mean(tf.reduce_min(image_attention, axis=1)))

        if options.text_regularizer_weight > 0.0:
          log_caption_attention = tf.log(
              tf.maximum(caption_attention, _SMALL_NUMBER))
          loss = tf.multiply(
              options.text_regularizer_weight,
              tf.reduce_mean(
                tf.reduce_sum(log_caption_attention * word_mask, axis=1)))
          tf.losses.add_loss(loss)
          tf.summary.scalar('loss/caption_attention_log_loss', loss)
          tf.summary.scalar('loss/caption_attention_max', 
              tf.reduce_mean(
                utils.masked_maximum(caption_attention, word_mask, dim=1)))
          tf.summary.scalar('loss/caption_attention_min', 
              tf.reduce_mean(
                utils.masked_minimum(caption_attention, word_mask, dim=1)))

        saliency_mask = self._calc_pairwise_similarity(
            image_feature=tf.expand_dims(image_attention, -1),
            text_feature=tf.expand_dims(caption_attention, -1),
            dropout_keep_prob=options.dropout_keep_prob,
            is_training=is_training)

        # Compute weighted sum.

        similarity = tf.reduce_sum(similarity * saliency_mask, axis=[1, 3])

        self.visualize(image, 
            tf.reshape(image_saliency, [-1, feature_height, feature_width]))
        tf.summary.histogram('image_saliency', image_saliency)
        tf.summary.histogram('text_saliency', caption_saliency)

      else:
        
        # Simple Global Average Pooling.

        similarity = tf.div(
            tf.reduce_sum(similarity, axis=[1, 3]),
            _SMALL_NUMBER + tf.cast(feature_width * feature_height *
              caption_lengths_gathered, tf.float32))

    predictions = {
      GAPPredictions.image_id: image_id,
      GAPPredictions.image_ids_gathered: image_ids_gathered,
      GAPPredictions.similarity: similarity,
    }
    return predictions

  def build_prediction(self, examples, 
      prediction_task=GAPPredictionTasks.similarity, **kwargs):
    """Builds tf graph for prediction.

    Args:
      examples: dict of input tensors keyed by name.
      prediction_task: the specific prediction task.

    Returns:
      predictions: dict of prediction results keyed by name.
    """

    if prediction_task == GAPPredictionTasks.similarity:
      return self._predict_similarity(examples)

    elif prediction_task == GAPPredictionTasks.image_saliency:
      return self._predict_image_saliency(examples)

    raise ValueError("Invalid prediction task %s" % (prediction_task))

  def visualize(self, image, saliency, 
      interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    """Visualizes images to tensorboard.

    Args:
      image: a [batch, height, width, channels] float tensor, in [0, 255].
      saliency: a [batch, feature_height, feature_width] float tensor.
    """
    (batch, height, width, channels) = utils.get_tensor_shape(image)

    image = image / 255.0
    heatmap = plotlib.convert_to_heatmap(saliency, normalize=True)
    heatmap = tf.image.resize_images(
        heatmap, [height, width], interpolation)

    heatmap = plotlib.gaussian_filter(heatmap, ksize=32)

    image = tf.concat([image, heatmap], axis=2)
    tf.summary.image("images", image, max_outputs=10)

  def build_loss(self, predictions, **kwargs):
    """Build tf graph to compute loss.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      loss_dict: dict of loss tensors keyed by name.
    """
    options = self._model_proto

    # Extracts tensors and shapes.

    (image_id, image_ids_gathered, similarity) = (
       predictions[GAPPredictions.image_id],
       predictions[GAPPredictions.image_ids_gathered],
       predictions[GAPPredictions.similarity])

    # Triplet loss.
    # Distance matrix, shape = [batch, num_captions_in_batch].

    with tf.name_scope(OperationNames.mine_in_batch_triplet):
      distance = 1.0 - similarity

      pos_mask = tf.cast(tf.equal(
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

        distance_an = tf.where(
            mask_condition, negatives_outside, negatives_inside)

      else:

        # Use the hardest.

        distance_an = utils.masked_minimum(distance, neg_mask)

    # Triplet loss.

    losses = tf.maximum(
        distance_ap - distance_an + options.triplet_loss_margin, 0)

    num_loss_examples = tf.count_nonzero(losses, dtype=tf.float32)
    loss = tf.div(
        tf.reduce_sum(losses), _SMALL_NUMBER + num_loss_examples,
        name="triplet_loss")

    return {'triplet_loss': loss}

  def build_evaluation(self, predictions, **kwargs):
    """Build tf graph to evaluate the model.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      eval_metric_ops: dict of metric results keyed by name. The values are the
        results of calling a metric function, namely a (metric_tensor, 
        update_op) tuple. see tf.metrics for details.
    """

    # Extracts tensors and shapes.

    (image_id, image_ids_gathered, similarity) = (
       predictions[GAPPredictions.image_id],
       predictions[GAPPredictions.image_ids_gathered],
       predictions[GAPPredictions.similarity])

    # Process retrieval on the in-batch eval dataset.

    with tf.name_scope(OperationNames.caption_retrieval):
      retrieved_index = tf.argmax(similarity, axis=1)
      predicted_alignment = tf.gather(
          image_ids_gathered, tf.argmax(similarity, axis=1))

    # Calculate accuracy.

    accuracy, update_op = tf.metrics.accuracy(image_id, predicted_alignment)

    return {'accuracy': (accuracy, update_op)}
