from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import nets_factory

from core import utils
from core import imgproc
from core import plotlib
from protos import cnn_pb2

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
    rgb_mean = [123.68, 116.779, 103.939]
    return image - tf.reshape(rgb_mean, [1, 1, 1, -1])

  raise ValueError('Invalid preprocess method {.}'.format(method))


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

  image_feature = slim.dropout(
      end_points[options.output_name],
      keep_prob=options.dropout_keep_prob,
      is_training=is_training)

  # Initialize from pre-trained checkpoint.

  if options.checkpoint_path:
    tf.train.init_from_checkpoint(
        options.checkpoint_path,
        assignment_map={"/": options.scope + "/"})
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
    vocabulary_list = [word.strip('\n') for word in fid.readlines()]
  return vocabulary_list


def gather_in_batch_captions(image_id, num_captions, caption_strings,
                             caption_lengths):
  """Gathers all of the in-batch captions into a caption batch.

  Args:
    image_id: image_id, a [batch] string tensor.
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


def build_proposal_saliency_fn(func_name, border_ratio, purity_weight,
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
      ymin, xmin = tf.maximum(ymin, 3), tf.maximum(xmin, 3)
      ymax, xmax = tf.minimum(ymax, tf.to_int64(n - 3)), tf.minimum(xmax, tf.to_int64(m - 3))

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
      tf.tile(
          tf.expand_dims(tf.range(batch, dtype=tf.int32), axis=-1), [1, k]),
      top_k_indices
  ], axis=-1)
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
