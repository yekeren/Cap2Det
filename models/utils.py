from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import nets_factory
from nets import vgg

from core import utils
from core import imgproc
from core import plotlib
from core import box_utils
#from protos import cnn_pb2
from object_detection.core.post_processing import batch_multiclass_non_max_suppression
from object_detection.builders.model_builder import _build_faster_rcnn_feature_extractor as build_faster_rcnn_feature_extractor
from pattern.en import pluralize, singularize
import numpy as np

_SMALL_NUMBER = 1e-10

slim = tf.contrib.slim


def preprocess_image(image, method='inception'):
  """Returns the preprocessed image.

  Args:
    image: a [batch, height, width, channels] float tensor, the values are 
      ranging from [0.0, 255.0].

  Returns:
    preproceed_image: a [batch, height, width, channels] float tensor.

  Raises:
    ValueError if method is invalid.
  """

  if "inception" == method:
    return image * 2.0 / 255.0 - 1.0

  elif "vgg" == method:
    rgb_mean = [123.68, 116.78, 103.94]
    return image - tf.reshape(rgb_mean, [1, 1, 1, -1])

  raise ValueError('Invalid preprocess method {.}'.format(method))


def vgg16_fc(image_feature, options, reuse=False, is_training=False):
  """Calculates vgg fc features.

  Args:
    feature_map: A [batch, feature_map_size, feature_map_size, feature_map_dims]
      float tensor.

  Returns:
    feature: A [batch, feature_dims] float tensor.
  """
  if not isinstance(options, cnn_pb2.CNN):
    raise ValueError('Invalid options.')

  net = image_feature

  with slim.arg_scope(vgg.vgg_arg_scope()):
    with tf.variable_scope(options.scope, reuse=reuse):
      with tf.variable_scope('vgg_16'):

        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(
            net,
            options.dropout_keep_prob,
            is_training=is_training and options.trainable,
            scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')

  # Initialize from pre-trained checkpoint.

  if options.checkpoint_path:
    tf.train.init_from_checkpoint(
        options.checkpoint_path, assignment_map={"/": options.scope + "/"})
  return net


def dilated_vgg16_conv(image, options, reuse=False, is_training=False):
  """Calculates dilated vgg16 feature based on options.

  Args:
    image: A [batch, height, width, channels] float tensor.
    options: A cnn_pb2.CNN instance.
    reuse: If True, reuse variables in the variable scope.
    is_training: If True, build the training graph.

  Returns:
    image_feature: A [batch, feature_height, feature_width, feature_dims] 
      float tensor.
  """
  if not isinstance(options, cnn_pb2.CNN):
    raise ValueError('Invalid options.')

  image = preprocess_image(image, 'vgg')

  with slim.arg_scope(vgg.vgg_arg_scope()):
    with tf.variable_scope(options.scope, reuse=reuse):
      with tf.variable_scope('vgg_16') as sc:
        net = slim.repeat(image, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')

        # Change to use atrous conv after removing the `pool4`.

        if not options.remove_pool4:
          net = slim.max_pool2d(net, [2, 2], scope='pool4')

        rate = options.dilate_rate
        net = slim.repeat(
            net,
            3,
            slim.conv2d,
            512, [3, 3],
            rate=options.dilate_rate,
            padding='SAME',
            scope='conv5')

        tf.logging.info('Feature map size is %s.', net.get_shape())

  # Initialize from pre-trained checkpoint.

  if options.checkpoint_path:
    tf.train.init_from_checkpoint(
        options.checkpoint_path, assignment_map={"/": options.scope + "/"})
  return net


def calc_cnn_feature(image, options, reuse=False, is_training=False):
  """Calculates CNN feature based on options.

  Args:
    image: A [batch, height, width, channels] float tensor.
    options: A cnn_pb2.CNN instance.
    reuse: If True, reuse variables in the variable scope.
    is_training: If True, build the training graph.

  Returns:
    image_feature: A [batch, feature_height, feature_width, feature_dims] 
      float tensor.
  """
  if not isinstance(options, cnn_pb2.CNN):
    raise ValueError('Invalid options.')

  # Preprocess the image.

  if options.preprocess_method == cnn_pb2.CNN.VGG:
    image = preprocess_image(image, 'vgg')
  elif options.preprocess_method == cnn_pb2.CNN.INCEPTION:
    image = preprocess_image(image, 'inception')
  else:
    raise ValueError('Invalid preprocess method.')

  # Call net_factory.

  with tf.variable_scope(options.scope, reuse=reuse):
    net_fn = nets_factory.get_network_fn(
        name=options.name,
        num_classes=None,
        weight_decay=options.weight_decay,
        is_training=is_training and options.trainable)
    _, end_points = net_fn(image)

  image_feature = end_points[options.output_name]
  image_feature = tf.reduce_mean(image_feature, [1, 2], name='AvgPool')
  image_feature = slim.dropout(
      image_feature,
      keep_prob=options.dropout_keep_prob,
      is_training=is_training)

  # Initialize from pre-trained checkpoint.

  if options.checkpoint_path:
    tf.train.init_from_checkpoint(
        options.checkpoint_path, assignment_map={"/": options.scope + "/"})
  return image_feature


def extract_image_feature(image,
                          name="vgg_16",
                          weight_decay=0.0,
                          output_name="vgg_16/fc7",
                          is_training=False):
  """Builds image model.

  Args:
    image: a [batch, height, width, channels] float tensor.
    name: name of the backbone CNN network.
    weight_decay: weight decay of the CNN network.
    output_name: name of the output tensor.
    is_training: if True, the training graph is built.

  Returns:
    output_tensor: a [batch, feature_height, feature_width, feature_dims] 
      float tensor.
  """
  net_fn = nets_factory.get_network_fn(
      name=name,
      num_classes=None,
      weight_decay=weight_decay,
      is_training=is_training)
  _, end_points = net_fn(image)

  if type(output_name) == str:
    return end_points[output_name]
  elif type(output_name) == list:
    return [end_points[name] for name in output_name]

  raise ValueError('Invalid output_name parameter.')


def read_vocabulary(filename):
  """Reads vocabulary list from file.

  Args:
    filename: path to the file storing vocabulary info.

  Returns:
    vocabulary_list: a list of string.
  """
  with tf.gfile.GFile(filename, "r") as fid:
    vocabulary_list = [line.strip('\n').split('\t')[0] for line in fid.readlines()]
  return vocabulary_list


def read_vocabulary_with_frequency(filename):
  """Reads vocabulary list from file.

  Args:
    filename: path to the file storing vocabulary info.

  Returns:
    vocabulary_list: a list of string.
  """
  with tf.gfile.GFile(filename, "r") as fid:
    vocabulary_list = [line.strip('\n').split('\t') for line in fid.readlines()]
    vocabulary_list = [(x[0], int(x[1])) for x in vocabulary_list]
  return vocabulary_list


def gather_in_batch_captions(image_id, num_captions, caption_strings,
                             caption_lengths):
  """Gathers all of the in-batch captions into a caption batch.

  Args:
    image_id: image_id, a [batch] int64 tensor.
    num_captions: number of captions of each example, a [batch] int tensor.
    caption_strings: caption data, a [batch, max_num_captions, 
      max_caption_length] string tensor.
    caption_lengths: length of each caption, a [batch, max_num_captions] int
      tensor.

  Returns:
    image_ids_gathered: associated image_id of each caption in the new batch, a
      [num_captions_in_batch] string tensor.
    caption_strings_gathered: caption data, a [num_captions_in_batch,
      max_caption_length] string tensor.
    caption_lengths_gathered: length of each caption, a [num_captions_in_batch]
      int tensor.
  """
  if not image_id.dtype in [tf.int32, tf.int64]:
    raise ValueError('The image_id has to be int32 or int64')

  (batch, max_num_captions,
   max_caption_length) = utils.get_tensor_shape(caption_strings)

  # caption_mask denotes the validity of each caption in the flattened batch.
  # caption_mask shape = [batch * max_num_captions],

  caption_mask = tf.sequence_mask(
      num_captions, maxlen=max_num_captions, dtype=tf.bool)
  caption_mask = tf.reshape(caption_mask, [-1])

  # image_id shape = [batch, max_num_captions].

  image_id = tf.tile(tf.expand_dims(image_id, axis=1), [1, max_num_captions])

  # Reshape the tensors to make their first dimensions to be [batch * max_num_captions].

  image_id_reshaped = tf.reshape(image_id, [-1])
  caption_strings_reshaped = tf.reshape(caption_strings,
                                        [-1, max_caption_length])
  caption_lengths_reshaped = tf.reshape(caption_lengths, [-1])

  # Apply the caption_mask.

  image_ids_gathered = tf.boolean_mask(image_id_reshaped, caption_mask)
  caption_strings_gathered = tf.boolean_mask(caption_strings_reshaped,
                                             caption_mask)
  caption_lengths_gathered = tf.boolean_mask(caption_lengths_reshaped,
                                             caption_mask)

  return image_ids_gathered, caption_strings_gathered, caption_lengths_gathered


def _get_expanded_box(box, img_h, img_w, border_ratio):
  """Gets expanded box.

  Args:
    box: a [..., 4] int tensor representing [ymin, xmin, ymax, xmax].
    img_h: image height.
    img_w: image width.
    border_ratio: width of the border in terms of percentage.

  Returns:
    expanded_box: a [..., 4] int tensor with border expanded.
  """
  ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
  (box_h, box_w) = ymax - ymin, xmax - xmin

  border_h = tf.cast(tf.cast(box_h, tf.float32) * border_ratio, tf.int64)
  border_w = tf.cast(tf.cast(box_w, tf.float32) * border_ratio, tf.int64)
  border_h = tf.maximum(border_h, 1)
  border_w = tf.maximum(border_w, 1)

  ymin_expanded = tf.maximum(ymin - border_h, 0)
  xmin_expanded = tf.maximum(xmin - border_w, 0)
  ymax_expanded = tf.minimum(ymax + border_h, tf.to_int64(img_h))
  xmax_expanded = tf.minimum(xmax + border_w, tf.to_int64(img_w))

  return tf.stack([ymin_expanded, xmin_expanded, ymax_expanded, xmax_expanded],
                  axis=-1)


def _get_shrinked_box(box, img_h, img_w, border_ratio):
  """Gets expanded box.

  Args:
    box: a [..., 4] int tensor representing [ymin, xmin, ymax, xmax].
    img_h: image height.
    img_w: image width.
    border_ratio: width of the border in terms of percentage.

  Returns:
    expanded_box: a [..., 4] int tensor with border expanded.
  """
  ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
  (box_h, box_w) = ymax - ymin, xmax - xmin

  border_h = tf.cast(tf.cast(box_h, tf.float32) * border_ratio, tf.int64)
  border_w = tf.cast(tf.cast(box_w, tf.float32) * border_ratio, tf.int64)
  border_h = tf.maximum(border_h, 1)
  border_w = tf.maximum(border_w, 1)

  mid_h = (ymin + ymax) // 2
  mid_w = (xmin + xmax) // 2

  ymin_expanded = tf.minimum(ymin + border_h, mid_h - 1)
  xmin_expanded = tf.minimum(xmin + border_w, mid_w - 1)
  ymax_expanded = tf.maximum(ymax - border_h, mid_h + 1)
  xmax_expanded = tf.maximum(xmax - border_w, mid_w + 1)

  return tf.stack([ymin_expanded, xmin_expanded, ymax_expanded, xmax_expanded],
                  axis=-1)


def _get_box_shape(box):
  """Gets the height and width of the box.

  Args:
    box: a [..., 4] int tensor representing [ymin, xmin, ymax, xmax].

  Returns:
    box_h: [...] box height.
    box_w: [...] box width.
  """
  ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
  return ymax - ymin, xmax - xmin


def _get_box_area(box):
  """Gets the box area.

  Args:
    box: a [..., 4] int tensor representing [ymin, xmin, ymax, xmax].

  Returns:
    area: a [...] int tensor representing area of the box.
  """
  box_h, box_w = _get_box_shape(box)
  return box_h * box_w


def build_proposal_saliency_fn(func_name,
                               border_ratio=None,
                               purity_weight=None,
                               **kwargs):
  """Builds and returns a callable to compute the proposal saliency.

  Args:
    func_name: name of the method.

  Returns:
    a callable that takes `score_map` and `box` as parameters.
  """
  if func_name == 'saliency_sum':
    return imgproc.calc_cumsum_2d

  if func_name == 'saliency_avg' or func_name == 'diba':

    def _cumsum_avg(score_map, box):
      ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
      area = tf.expand_dims((ymax - ymin) * (xmax - xmin), axis=-1)
      return tf.div(
          imgproc.calc_cumsum_2d(score_map, box),
          tf.maximum(_SMALL_NUMBER, tf.cast(area, tf.float32)))

    return _cumsum_avg

  if func_name == 'wei':

    def _cumsum_gradient(score_map, box):
      b, n, m, c = utils.get_tensor_shape(score_map)
      _, p, _ = utils.get_tensor_shape(box)

      # Leave a border for the image border.
      ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
      ymin, xmin = tf.maximum(ymin, 2), tf.maximum(xmin, 2)
      ymax, xmax = tf.minimum(ymax, tf.to_int64(n - 2)), tf.minimum(
          xmax, tf.to_int64(m - 2))

      box = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
      box_exp = _get_expanded_box(
          box, img_h=n, img_w=m, border_ratio=border_ratio)

      box_list = [box, box_exp]

      area_list = [tf.cast(_get_box_area(b), tf.float32) for b in box_list]
      cumsum = imgproc.calc_cumsum_2d(score_map, tf.concat(box_list, axis=1))
      cumsum_list = [
          cumsum[:, i * p:(i + 1) * p, :] for i in range(len(box_list))
      ]

      # The main box has to be valid, including the four shrinked boxes.
      assert_op = tf.Assert(
          tf.reduce_all(tf.greater(area_list[0], 0)),
          ["Check area of the main box failed:", area_list[0]])

      with tf.control_dependencies([assert_op]):
        border_area = area_list[1] - area_list[0]
        border_cumsum = cumsum_list[1] - cumsum_list[0]

        border_avg = tf.div(
            border_cumsum,
            tf.maximum(_SMALL_NUMBER, tf.expand_dims(border_area, axis=-1)))
        box_avg = tf.div(
            cumsum_list[0],
            tf.maximum(_SMALL_NUMBER, tf.expand_dims(area_list[0], axis=-1)))

        return purity_weight * box_avg - border_avg

    return _cumsum_gradient

  if func_name == 'saliency_grad':
    assert False

    def _cumsum_gradient(score_map, box):
      b, n, m, c = utils.get_tensor_shape(score_map)
      _, p, _ = utils.get_tensor_shape(box)

      expanded_box = _get_expanded_box(
          box, img_h=n, img_w=m, border_ratio=border_ratio)

      (box_h, box_w) = _get_box_shape(box)
      (expanded_box_h, expanded_box_w) = _get_box_shape(expanded_box)

      cumsum = imgproc.calc_cumsum_2d(score_map,
                                      tf.concat([box, expanded_box], axis=1))

      area = tf.expand_dims(tf.cast(box_h * box_w, tf.float32), axis=-1)
      area_border = tf.expand_dims(
          tf.cast(expanded_box_h * expanded_box_w - box_h * box_w, tf.float32),
          axis=-1)

      avg_val = tf.div(cumsum[:, :p, :], tf.maximum(_SMALL_NUMBER, area))
      avg_val_in_border = tf.div(cumsum[:, p:, :] - cumsum[:, :p, :],
                                 tf.maximum(_SMALL_NUMBER, area_border))

      return avg_val - avg_val_in_border

    return _cumsum_gradient

  if func_name == 'saliency_grad_v2':
    assert False

    def _cumsum_gradient(score_map, box):
      b, n, m, c = utils.get_tensor_shape(score_map)
      _, p, _ = utils.get_tensor_shape(box)

      box_exp = _get_expanded_box(
          box, img_h=n, img_w=m, border_ratio=border_ratio)
      box_shr = _get_shrinked_box(
          box, img_h=n, img_w=m, border_ratio=border_ratio)

      ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
      ymin_exp, xmin_exp, ymax_exp, xmax_exp = tf.unstack(box_exp, axis=-1)
      ymin_shr, xmin_shr, ymax_shr, xmax_shr = tf.unstack(box_shr, axis=-1)

      box_list = [
          box,
          tf.stack([ymin, xmin, ymax, xmin_shr], axis=-1),
          tf.stack([ymin, xmax_shr, ymax, xmax], axis=-1),
          tf.stack([ymin, xmin, ymin_shr, xmax], axis=-1),
          tf.stack([ymax_shr, xmin, ymax, xmax], axis=-1),
          tf.stack([ymin, xmin_exp, ymax, xmin], axis=-1),
          tf.stack([ymin, xmax, ymax, xmax_exp], axis=-1),
          tf.stack([ymin_exp, xmin, ymin, xmax], axis=-1),
          tf.stack([ymax, xmin, ymax_exp, xmax], axis=-1)
      ]

      area_list = [tf.cast(_get_box_area(b), tf.float32) for b in box_list]
      cumsum = imgproc.calc_cumsum_2d(score_map, tf.concat(box_list, axis=1))
      cumsum_list = [
          cumsum[:, i * p:(i + 1) * p, :] for i in range(len(box_list))
      ]

      # Compute the averaged cumsum inside each box.
      cumsum_avg_list = [
          tf.div(
              cumsum_list[i],
              tf.expand_dims(tf.maximum(_SMALL_NUMBER, area_list[i]), axis=-1))
          for i in range(len(box_list))
      ]

      # The main box has to be valid, including the four shrinked boxes.
      assert_op = tf.Assert(
          tf.reduce_all(tf.greater(tf.stack(area_list[:5], axis=-1), 0)), [
              "Check area of the main box failed:",
              tf.stack(area_list[:5], axis=-1)
          ])

      with tf.control_dependencies([assert_op]):
        # The expanded box can have ZERO area.
        grad_list = []
        for i in [1, 2, 3, 4]:
          area_mask = tf.tile(
              tf.expand_dims(tf.greater(area_list[i + 4], 0), axis=-1),
              [1, 1, c])
          grad = tf.where(
              area_mask,
              x=cumsum_avg_list[i] - cumsum_avg_list[i + 4],
              y=tf.ones_like(cumsum_avg_list[i], dtype=tf.float32))
          #grad = cumsum_avg_list[i] - cumsum_avg_list[i + 4]
          grad_list.append(grad)

        return purity_weight * cumsum_avg_list[0] + tf.reduce_min(
            tf.stack(grad_list, axis=-1), axis=-1)

    return _cumsum_gradient

  if func_name == 'saliency_grad_v3':

    def _cumsum_gradient(score_map, box):
      b, n, m, c = utils.get_tensor_shape(score_map)
      _, p, _ = utils.get_tensor_shape(box)

      n, m = tf.cast(n, tf.int64), tf.cast(m, tf.int64)

      # Leave a border for the image border.
      ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
      ymin, xmin = tf.maximum(ymin, 3), tf.maximum(xmin, 3)
      ymax, xmax = tf.minimum(ymax, n - 3), tf.minimum(xmax, m - 3)
      box = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

      box_exp = _get_expanded_box(
          box, img_h=n, img_w=m, border_ratio=border_ratio)
      box_shr = _get_shrinked_box(
          box, img_h=n, img_w=m, border_ratio=border_ratio)

      ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
      ymin_exp, xmin_exp, ymax_exp, xmax_exp = tf.unstack(box_exp, axis=-1)
      ymin_shr, xmin_shr, ymax_shr, xmax_shr = tf.unstack(box_shr, axis=-1)

      box_list = [
          box,
          tf.stack([ymin, xmin, ymax, xmin_shr], axis=-1),
          tf.stack([ymin, xmax_shr, ymax, xmax], axis=-1),
          tf.stack([ymin, xmin, ymin_shr, xmax], axis=-1),
          tf.stack([ymax_shr, xmin, ymax, xmax], axis=-1),
          tf.stack([ymin, xmin_exp, ymax, xmin], axis=-1),
          tf.stack([ymin, xmax, ymax, xmax_exp], axis=-1),
          tf.stack([ymin_exp, xmin, ymin, xmax], axis=-1),
          tf.stack([ymax, xmin, ymax_exp, xmax], axis=-1)
      ]

      area_list = [tf.cast(_get_box_area(b), tf.float32) for b in box_list]
      cumsum = imgproc.calc_cumsum_2d(score_map, tf.concat(box_list, axis=1))
      cumsum_list = [
          cumsum[:, i * p:(i + 1) * p, :] for i in range(len(box_list))
      ]

      # Compute the averaged cumsum inside each box.
      cumsum_avg_list = [
          tf.div(
              cumsum_list[i],
              tf.expand_dims(tf.maximum(_SMALL_NUMBER, area_list[i]), axis=-1))
          for i in range(len(box_list))
      ]

      # The main box has to be valid, including the four shrinked boxes.
      assert_op = tf.Assert(
          tf.reduce_all(tf.greater(tf.stack(area_list, axis=-1), 0)),
          ["Check area of the main box failed:", area_list[0]])

      with tf.control_dependencies([assert_op]):
        grad_list = []
        for i in [1, 2, 3, 4]:
          grad = cumsum_avg_list[i] - cumsum_avg_list[i + 4]
          grad_list.append(grad)

        return purity_weight * cumsum_avg_list[0] + tf.reduce_min(
            tf.stack(grad_list, axis=-1), axis=-1)

    return _cumsum_gradient

  raise ValueError('Invalid func_name {}'.format(func_name))


def get_top_k_boxes_and_scores(boxes, box_scores, box_labels=None, k=1):
  """Gets the top-k boxes and scores.

  Args:
    boxes: A [batch, number_of_boxes, 4] float tensor.
    box_scores: A [batch, number_of_boxes] float tensor.
    box_labels: A [batch, number_of_boxes] string tensor.

  Returns:
    top_k_boxes: A [batch, top_k, 4] float tensor.
    top_k_scores: A [batch, top_k] float tensor.
  """
  batch = boxes.get_shape()[0].value

  top_k_scores, top_k_indices = tf.nn.top_k(box_scores, k)
  top_k_indices = tf.stack([
      tf.tile(tf.expand_dims(tf.range(batch, dtype=tf.int32), axis=-1), [1, k]),
      top_k_indices
  ],
                           axis=-1)
  top_k_boxes = tf.gather_nd(boxes, top_k_indices)
  top_k_labels = None
  if box_labels is not None:
    top_k_labels = tf.gather_nd(box_labels, top_k_indices)
  return top_k_boxes, top_k_scores, top_k_labels


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
  return net


def visl_proposals(image,
                   num_proposals,
                   proposals,
                   top_k=100,
                   height=224,
                   width=224,
                   name='proposals'):
  """Visualize proposal results to the tensorboard.

  Args:
    image: A [batch, height, width, channels] float tensor, 
      ranging from 0 to 255.
    num_proposals: A [batch] int tensor.
    proposals: A [batch, max_num_proposals, 4] float tensor.
    height: Height of the visualized image.
    width: Width of the visualized image.
  """
  with tf.name_scope('visl_proposals'):
    if height is not None and width is not None:
      image = tf.image.resize_images(image, [height, width])
    image = tf.cast(image, tf.uint8)
    image = plotlib.draw_rectangles(
        image, boxes=proposals[:, :top_k, :], color=plotlib.RED, fontscale=1.0)
  tf.summary.image(name, image, max_outputs=10)


def visl_detections(image,
                    num_detections,
                    detection_boxes,
                    detection_scores,
                    detection_classes,
                    name='visl_detection'):
  with tf.name_scope('visl_proposals'):
    image = tf.cast(image, tf.uint8)

    image = plotlib.draw_rectangles_v2(
        image,
        total=num_detections,
        boxes=detection_boxes,
        scores=detection_scores,
        labels=detection_classes,
        color=plotlib.RED,
        fontscale=0.8)

  tf.summary.image(name, image, max_outputs=10)


def visl_proposals_top_k(image,
                         num_proposals,
                         proposals,
                         proposal_scores,
                         proposal_labels=None,
                         top_k=5,
                         threshold=0.01,
                         height=224,
                         width=224,
                         name='midn'):
  """Visualize top proposal results to the tensorboard.

  Args:
    image: A [batch, height, width, channels] float tensor, 
      ranging from 0 to 255.
    num_proposals: A [batch] int tensor.
    proposals: A [batch, max_num_proposals, 4] float tensor.
    proposal_scores: A [batch, max_num_proposals] float tensor.
    proposal_labels: A [batch, max_num_proposals] float tensor.
    height: Height of the visualized image.
    width: Width of the visualized image.
  """
  with tf.name_scope('visl_proposals'):
    image = tf.image.resize_images(image, [height, width])
    image = tf.cast(image, tf.uint8)

    (top_k_boxes, top_k_scores, top_k_labels) = get_top_k_boxes_and_scores(
        proposals, proposal_scores, proposal_labels, k=top_k)

    top_k_scores = tf.where(top_k_scores > threshold, top_k_scores,
                            -9999.0 * tf.ones_like(top_k_scores))
    image = plotlib.draw_rectangles(
        image,
        boxes=top_k_boxes,
        scores=top_k_scores,
        labels=top_k_labels,
        color=plotlib.RED,
        fontscale=1.0)
  tf.summary.image(name, image, max_outputs=10)


def post_process(boxes,
                 scores,
                 score_thresh=1e-6,
                 iou_thresh=0.5,
                 max_size_per_class=100,
                 max_total_size=300):
  """Applies post process to get the final detections.

  Args:
    boxes: A [batch_size, num_anchors, q, 4] float32 tensor containing
      detections. If `q` is 1 then same boxes are used for all classes
        otherwise, if `q` is equal to number of classes, class-specific boxes
        are used.
    scores: A [batch_size, num_anchors, num_classes] float32 tensor containing
      the scores for each of the `num_anchors` detections. The scores have to be
      non-negative when use_static_shapes is set True.
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (new boxes that have high IOU overlap
      with previously selected boxes are removed).
    max_size_per_class: maximum number of retained boxes per class.
    max_total_size: maximum number of boxes retained over all classes. By
      default returns all boxes retained after capping boxes per class.

Returns:
  num_detections: A [batch_size] int32 tensor indicating the number of
    valid detections per batch item. Only the top num_detections[i] entries in
    nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
    entries are zero paddings.
  nmsed_boxes: A [batch_size, max_detections, 4] float32 tensor
    containing the non-max suppressed boxes.
  nmsed_scores: A [batch_size, max_detections] float32 tensor containing
    the scores for the boxes.
  nmsed_classes: A [batch_size, max_detections] float32 tensor
    containing the class for boxes.
  """
  boxes = tf.expand_dims(boxes, axis=2)
  (nmsed_boxes, nmsed_scores, nmsed_classes, _, _,
   num_detections) = batch_multiclass_non_max_suppression(
       boxes,
       scores,
       score_thresh=score_thresh,
       iou_thresh=iou_thresh,
       max_size_per_class=max_size_per_class,
       max_total_size=max_total_size)
  return num_detections, nmsed_boxes, nmsed_scores, nmsed_classes + 1


def calc_oicr_loss(labels,
                   num_proposals,
                   proposals,
                   scores_0,
                   scores_1,
                   scope,
                   iou_threshold=0.5):
  """Calculates the NOD loss at refinement stage `i`.

  Args:
    labels: A [batch, num_classes] float tensor.
    num_proposals: A [batch] int tensor.
    proposals: A [batch, max_num_proposals, 4] float tensor.
    scores_0: A [batch, max_num_proposal, 1 + num_classes] float tensor, 
      representing the proposal score at `k-th` refinement.
    scores_1: A [batch, max_num_proposal, 1 + num_classes] float tensor,
      representing the proposal score at `(k+1)-th` refinement.

  Returns:
    oicr_cross_entropy_loss: a scalar float tensor.
  """
  with tf.name_scope(scope):
    (batch, max_num_proposals,
     num_classes_plus_one) = utils.get_tensor_shape(scores_0)
    num_classes = num_classes_plus_one - 1

    # For each class, look for the most confident proposal.
    #   proposal_ind shape = [batch, num_classes].

    proposal_mask = tf.sequence_mask(
        num_proposals, maxlen=max_num_proposals, dtype=tf.float32)
    proposal_ind = utils.masked_argmax(
        scores_0[:, :, 1:], tf.expand_dims(proposal_mask, axis=-1), dim=1)

    # Deal with the most confident proposal per each class.
    #   Unstack the `proposal_ind`, `labels`.
    #   proposal_labels shape = [batch, max_num_proposals, num_classes].

    proposal_labels = []
    indices_0 = tf.range(batch, dtype=tf.int64)
    for indices_1, label_per_class in zip(
        tf.unstack(proposal_ind, axis=-1), tf.unstack(labels, axis=-1)):

      # Gather the most confident proposal for the class.
      #   confident_proosal shape = [batch, 4].

      indices = tf.stack([indices_0, indices_1], axis=-1)
      confident_proposal = tf.gather_nd(proposals, indices)

      # Get the Iou from all the proposals to the most confident proposal.
      #   iou shape = [batch, max_num_proposals].

      confident_proposal_tiled = tf.tile(
          tf.expand_dims(confident_proposal, axis=1), [1, max_num_proposals, 1])
      iou = box_utils.iou(
          tf.reshape(proposals, [-1, 4]),
          tf.reshape(confident_proposal_tiled, [-1, 4]))
      iou = tf.reshape(iou, [batch, max_num_proposals])

      # Filter out irrelevant predictions using image-level label.

      target = tf.to_float(tf.greater_equal(iou, iou_threshold))
      target = tf.where(label_per_class > 0, x=target, y=tf.zeros_like(target))
      proposal_labels.append(target)

    proposal_labels = tf.stack(proposal_labels, axis=-1)

    # Add background targets, and normalize the sum value to 1.0.
    #   proposal_labels shape = [batch, max_num_proposals, 1 + num_classes].

    bkg = tf.logical_not(tf.reduce_sum(proposal_labels, axis=-1) > 0)
    proposal_labels = tf.concat(
        [tf.expand_dims(tf.to_float(bkg), axis=-1), proposal_labels], axis=-1)

    proposal_labels = tf.div(
        proposal_labels, tf.reduce_sum(proposal_labels, axis=-1, keepdims=True))

    assert_op = tf.Assert(
        tf.reduce_all(
            tf.abs(tf.reduce_sum(proposal_labels, axis=-1) - 1) < 1e-6),
        ["Probabilities not sum to ONE", proposal_labels])

    # Compute the loss.

    with tf.control_dependencies([assert_op]):
      losses = tf.nn.softmax_cross_entropy_with_logits(
          labels=tf.stop_gradient(proposal_labels), logits=scores_1)
      oicr_cross_entropy_loss = tf.reduce_mean(
          utils.masked_avg(data=losses, mask=proposal_mask, dim=1))

  return oicr_cross_entropy_loss


def calc_pairwise_similarity(feature_a,
                             feature_b,
                             dropout_keep_prob=1.0,
                             l2_normalize=True,
                             is_training=False):
  """Computes the similarity between the two modality.

  Args:
    feature_a: A [batch_a, feature_dims] float tensor.
    feature_b: A [batch_b, feature_dims] float tensor.

  Returns:
    A [batch_a, batch_b] similarity matrix.
  """
  if l2_normalize:
    feature_a = tf.nn.l2_normalize(feature_a, axis=-1)
    feature_b = tf.nn.l2_normalize(feature_b, axis=-1)

  feature_a = tf.expand_dims(feature_a, axis=1)
  feature_b = tf.expand_dims(feature_b, axis=0)
  dot_product = tf.multiply(feature_a, feature_b)
  return tf.reduce_sum(dot_product, axis=-1)


def read_synonyms(filename):
  """Reads synonyms dict.

  Args:
    filename: Path to the synonyms dict.
  Returns:
    A dict mapping from synonym word to its original form.
  """
  data = {}
  with open(filename, 'r') as fid:
    for line in fid.readlines():
      word, synonyms = line.strip('\n').split('\t')
      for w in synonyms.split(','):
        data[w] = word
        data[pluralize(w)] = word
        data[singularize(w)] = word
  return data


# For coco synonyms mapping.
class_synonyms = {
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


def substitute_class_names(vocabulary_list):
  return [class_synonyms.get(x, x) for x in vocabulary_list]


def extract_frcnn_feature(inputs,
                          num_proposals,
                          proposals,
                          options,
                          is_training=False):
  """Extracts Fast-RCNN feature from image.

  Args:
    feature_extractor: An FRCNN feature extractor instance.
    inputs: A [batch, height, width, channels] float tensor.
    num_proposals: A [batch] int tensor.
    proposals: A [batch, max_num_proposals, 4] float tensor.
    options:
    is_training:

  Returns:
    proposal_features: A [batch, max_num_proposals, feature_dims] float 
      tensor.
  """
  feature_extractor = build_faster_rcnn_feature_extractor(
      options.feature_extractor, is_training, options.inplace_batchnorm_update)

  # Extract `features_to_crop` from the original image.
  #   shape = [batch, feature_height, feature_width, feature_depth].

  preprocessed_inputs = feature_extractor.preprocess(inputs)

  (features_to_crop, _) = feature_extractor.extract_proposal_features(
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
  ) = feature_extractor.extract_box_classifier_features(
      flattened_proposal_features_maps, scope='second_stage_feature_extraction')

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


def load_glove_data(filename):
  """Loads the glove data in .txt format.

  Args:
    filename: Path to the GloVe data.

  Returns:
    word2vec: A dict mapping from words to their embedding vectors.
    dims: Dimensions of the word embedding.
  """
  with open(filename, 'r', encoding='utf-8') as fp:
    lines = fp.readlines()

  dims = len(lines[0].strip('\n').split()) - 1

  word2vec = {}
  for line_index, line in enumerate(lines):
    items = line.strip('\n').split()
    word, vec = items[0], np.array([float(v) for v in items[1:]])
    assert vec.shape[0] == dims

    word2vec[word] = vec
    if line_index % 10000== 0:
      tf.logging.info('On load GloVe %s/%s', line_index, len(lines))
  return word2vec, dims


