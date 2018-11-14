from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from nets import nets_factory

from models.model_base import ModelBase
from protos import voc_model_pb2

from core import imgproc
from core import utils
from core import plotlib
from core.standard_fields import OperationNames
from core.standard_fields import InputDataFields
from core.standard_fields import VOCVariableScopes
from core.standard_fields import VOCPredictions
from core.standard_fields import VOCPredictionTasks
from core.training_utils import build_hyperparams
from models import utils as model_utils

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


class Model(ModelBase):
  """VOC model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of voc_model_pb2.VOCModel
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, voc_model_pb2.VOCModel):
      raise ValueError('The model_proto has to be an instance of VOCModel.')

  def get_scaffold(self):
    """Returns scaffold object used to initialize variables.

    Returns:
      a tf.train.Scaffold instance or None by default.
    """

    def _init_fn(unused_scaffold, sess):
      """Function for initialization.

      Args:
        sess: a tf.Session instance.
      """
      tf.logging.info("Initialize using scaffold.")

    scaffold = tf.train.Scaffold(init_fn=_init_fn)
    return scaffold

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      a list of model variables or None by default.
    """
    options = self._model_proto

    # Filter out variables that are not trainable.
    variables_to_train = []
    for var in tf.trainable_variables():
      if not options.cnn_trainable:
        if VOCVariableScopes.cnn in var.op.name:
          tf.logging.info("Freeze cnn parameter %s.", var.op.name)
          continue
      variables_to_train.append(var)

    # Print trainable variables.

    tf.logging.info('*' * 128)
    for var in variables_to_train:
      tf.logging.info("Model variables: %s.", var.op.name)

    return variables_to_train

  def _preprocess(self, image):
    """Returns the preprocessed image.

    Args:
      image: a [batch, height, width, channels] float tensor, the values are 
        ranging from [0.0, 255.0].

    Returns:
      preproceed_image: a [batch, height, width, channels] float tensor.
    """
    options = self._model_proto
    if "inception" == options.preprocessing_method:
      return image * 2.0 / 255.0 - 1.0

    elif "vgg" == options.preprocessing_method:
      rgb_mean = [123.68, 116.779, 103.939]
      return image - tf.reshape(rgb_mean, [1, 1, 1, -1])

    raise ValueError('Invalid preprocessing method {}'.format(
        options.preprocessing_method))

  def _encode_images(self,
                     image,
                     cnn_name="mobilenet_v2",
                     cnn_trainable=False,
                     cnn_weight_decay=1e-4,
                     cnn_feature_map="layer_18/output",
                     cnn_dropout_keep_prob=1.0,
                     cnn_checkpoint=None,
                     cnn_scope="CNN",
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
      is_training: if True, training graph is built.

    Returns:
      feature_map: a [batch, feature_height, feature_width, common_dimensions] 
        float tensor.
    """
    # Preprocess and extract CNN feature.

    image = self._preprocess(image)
    net_fn = nets_factory.get_network_fn(
        name=cnn_name,
        num_classes=None,
        weight_decay=cnn_weight_decay,
        is_training=cnn_trainable and is_training)

    with tf.variable_scope(cnn_scope):
      _, end_points = net_fn(image)
    feature_map = end_points[cnn_feature_map]

    feature_map = tf.contrib.layers.dropout(
        feature_map, keep_prob=cnn_dropout_keep_prob, is_training=is_training)

    # Load pre-trained model from checkpoint.

    if not cnn_checkpoint:
      tf.logging.warning("Pre-trained checkpoint path is not provided!")
    else:
      tf.train.init_from_checkpoint(
          cnn_checkpoint, assignment_map={"/": VOCVariableScopes.cnn + "/"})

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

  def _encode_labels(self,
                     num_captions,
                     caption_strings,
                     caption_lengths,
                     vocabulary_list,
                     is_training=False):
    """Encodes labels.

    Args:
      num_captions: a [batch] int tensor.
      caption_strings: a [batch, num_captions, max_caption_len] string tensor.
      caption_lengths: a [batch, num_captions] int tensor.
      vocabulary_list: a list of words of length ``num_classes''''.
      is_training: if True, training graph is built.

    Returns:
      classes: a [batch, num_classes] int tensor.
    """
    with tf.name_scope('encode_labels'):
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
      classes = tf.cast(indicator[:, :-1] > 0, tf.int64)
      tf.summary.histogram('num_gt_boxes_per_image', caption_length)
      tf.summary.histogram('num_gt_labels_per_image',
                           tf.reduce_sum(classes, axis=-1))

    return classes

  def _project_images(self,
                      feature_map,
                      common_dimensions=300,
                      scope="image_proj",
                      hyperparams=None,
                      is_training=False):
    """Adds additional 1x1 conv layer to project image features.

    Args:
      feature_map: [batch, feature_height, feature_width, feature_depth] float
        tensor, which is the CNN output.
      common_dimensions: depth of the image embedding.
      scope: variable scope of the projection layer.
      hyperparams: an instance of hyperparams_pb2.Hyperparams, used for the
        conv2d projection layer.
      is_training: if True, training graph is built.
    """
    with slim.arg_scope(build_hyperparams(hyperparams, is_training)):
      with tf.variable_scope(scope):
        feature_map = tf.contrib.layers.conv2d(
            inputs=feature_map,
            num_outputs=common_dimensions,
            kernel_size=[1, 1],
            activation_fn=None)
    return feature_map

  def _predict_labels(self, examples):
    """Builds tf graph for prediction.

    Args:
      examples: dict of input tensors keyed by name.

    Returns:
      predictions: dict of prediction results keyed by name.
    """
    options = self._model_proto
    is_training = self._is_training

    # Load the vocabulary.
    vocabulary_list = self._read_vocabulary(options.vocabulary_file)
    tf.logging.info("Read a vocabulary with %i words.", len(vocabulary_list))

    # Extract input data fields.

    (image, image_id, num_captions, caption_strings,
     caption_lengths) = (examples[InputDataFields.image],
                         examples[InputDataFields.image_id],
                         examples[InputDataFields.num_captions],
                         examples[InputDataFields.caption_strings],
                         examples[InputDataFields.caption_lengths])

    # Encode image use CNN.

    image_feature = self._encode_images(
        image,
        cnn_name=options.cnn_name,
        cnn_trainable=options.cnn_trainable,
        cnn_weight_decay=options.cnn_weight_decay,
        cnn_feature_map=options.cnn_feature_map,
        cnn_dropout_keep_prob=options.cnn_dropout_keep_prob,
        cnn_checkpoint=options.cnn_checkpoint,
        cnn_scope=VOCVariableScopes.cnn,
        is_training=is_training)

    # Predict class activation map, shape =
    #   [batch, feature_height * feature_width, num_classes].

    with tf.name_scope(OperationNames.image_model):
      class_act_map = self._project_images(
          image_feature,
          common_dimensions=len(vocabulary_list),
          scope=VOCVariableScopes.image_proj,
          hyperparams=options.image_proj_hyperparams,
          is_training=is_training)

      if options.use_gmp:
        logits = tf.reduce_max(class_act_map, axis=[1, 2])
      else:  # Use GAP by default
        logits = tf.reduce_mean(class_act_map, axis=[1, 2])

      tf.summary.histogram('class_act_map', class_act_map)

    # Encode labels, shape=[batch, num_classes].

    class_labels = self._encode_labels(num_captions, caption_strings,
                                       caption_lengths, vocabulary_list)

    # visualize

    image_vis = tf.cast(image, tf.uint8)
    image_vis = plotlib.draw_caption(
        image_vis,
        tf.reduce_join(caption_strings[:, 0, :], axis=-1, separator=','),
        org=(5, 5),
        fontscale=1.0,
        color=(255, 0, 0),
        thickness=1)
    image_vis = plotlib.draw_caption(
        image_vis,
        tf.gather(vocabulary_list, tf.argmax(logits, axis=-1)),
        org=(5, 25),
        fontscale=1.0,
        color=(255, 0, 0),
        thickness=1)

    class_act_map_list = []
    batch_size, height, width, _ = utils.get_tensor_shape(image_vis)
    for i, x in enumerate(tf.unstack(class_act_map, axis=-1)):
      x = plotlib.convert_to_heatmap(x, normalize=True, normalize_to=[-4, 4])
      #x = tf.image.resize_images(x, [height, width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      x = tf.image.resize_images(x, [height, width])
      x = imgproc.gaussian_filter(x, ksize=32)
      x = tf.image.convert_image_dtype(x, tf.uint8)
      x = plotlib.draw_caption(
          x,
          tf.tile(tf.expand_dims(vocabulary_list[i], axis=0), [batch_size]),
          org=(5, 5),
          fontscale=1.0,
          color=(255, 0, 0),
          thickness=1)
      class_act_map_list.append(x)
    tf.summary.image(
        "image",
        tf.concat([image_vis] + class_act_map_list, axis=2),
        max_outputs=1)

    predictions = {
        VOCPredictions.image_id: image_id,
        VOCPredictions.class_labels: class_labels,
        VOCPredictions.class_act_map: class_act_map,
        VOCPredictions.per_class_logits: logits,
    }

    return predictions

  def build_prediction(self,
                       examples,
                       prediction_task=VOCPredictionTasks.class_labels,
                       **kwargs):
    """Builds tf graph for prediction.

    Args:
      examples: dict of input tensors keyed by name.
      prediction_task: the specific prediction task.

    Returns:
      predictions: dict of prediction results keyed by name.
    """

    if prediction_task == VOCPredictionTasks.class_labels:
      return self._predict_labels(examples)

    raise ValueError("Invalid prediction task %s" % (prediction_task))

  def build_loss(self, predictions, **kwargs):
    """Build tf graph to compute loss.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      loss_dict: dict of loss tensors keyed by name.
    """
    options = self._model_proto

    with tf.name_scope('Losses'):
      (image_id, class_act_map, class_labels,
       per_class_logits) = (predictions[VOCPredictions.image_id],
                            predictions[VOCPredictions.class_act_map],
                            predictions[VOCPredictions.class_labels],
                            predictions[VOCPredictions.per_class_logits])

      # logits = tf.reduce_max(class_act_map, axis=[1, 2])
      logits = per_class_logits

      class_labels = tf.cast(class_labels, tf.float32)

      if options.use_sigmoid_instead_of_softmax:

        # sigmoid-cross-entropy.

        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=class_labels, logits=logits)
      else:

        # softmax-cross-entropy.

        class_labels = tf.div(
            class_labels, tf.reduce_sum(class_labels, axis=-1, keepdims=True))
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=class_labels, logits=logits)

    return {'cross_entropy_loss': tf.reduce_mean(losses)}

  def build_evaluation(self, predictions, **kwargs):
    """Build tf graph to evaluate the model.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      eval_metric_ops: dict of metric results keyed by name. The values are the
        results of calling a metric function, namely a (metric_tensor, 
        update_op) tuple. see tf.metrics for details.
    """
    with tf.name_scope('Evaluation'):
      (image_id, class_act_map, class_labels,
       per_class_logits) = (predictions[VOCPredictions.image_id],
                            predictions[VOCPredictions.class_act_map],
                            predictions[VOCPredictions.class_labels],
                            predictions[VOCPredictions.per_class_logits])

      # logits = tf.reduce_max(class_act_map, axis=[1, 2])
      logits = per_class_logits

      metrics = {}
      metric, update_op = tf.metrics.accuracy(
          labels=tf.argmax(class_labels, axis=-1),
          predictions=tf.argmax(logits, axis=-1))
      metrics.update({'accuracy': (metric, update_op)})

    return metrics
