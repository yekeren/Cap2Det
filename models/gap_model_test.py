from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from models import gap_model
from protos import gap_model_pb2
from protos import hyperparams_pb2

tf.logging.set_verbosity(tf.logging.INFO)


class GAPModelTest(tf.test.TestCase):

  def test_encode_images(self):
    model_proto = gap_model_pb2.GAPModel()
    model = gap_model.Model(model_proto, is_training=False)

    # MobilenetV1, 224x224 input.

    tf.reset_default_graph()

    image = tf.random_uniform(shape=[32, 224, 224, 3])
    feature_map = model._encode_images(
        image,
        cnn_name="mobilenet_v1",
        cnn_feature_map="Conv2d_13_pointwise",
        is_training=False)
    self.assertAllEqual(feature_map.get_shape().as_list(), [32, 7, 7, 1024])

    # MobilenetV1, 448x448 input.

    tf.reset_default_graph()

    image = tf.random_uniform(shape=[32, 448, 448, 3])
    feature_map = model._encode_images(
        image,
        cnn_name="mobilenet_v1",
        cnn_feature_map="Conv2d_13_pointwise",
        is_training=False)
    self.assertAllEqual(feature_map.get_shape().as_list(), [32, 14, 14, 1024])

    # MobilenetV2, 224x224 input.

    tf.reset_default_graph()

    image = tf.random_uniform(shape=[32, 224, 224, 3])
    feature_map = model._encode_images(
        image,
        cnn_name="mobilenet_v2",
        cnn_feature_map="layer_18/output",
        is_training=False)
    self.assertAllEqual(feature_map.get_shape().as_list(), [32, 7, 7, 320])

    # MobilenetV2, 448x448 input.

    tf.reset_default_graph()

    image = tf.random_uniform(shape=[32, 448, 448, 3])
    feature_map = model._encode_images(
        image,
        cnn_name="mobilenet_v2",
        cnn_feature_map="layer_18/output",
        is_training=False)
    self.assertAllEqual(feature_map.get_shape().as_list(), [32, 14, 14, 320])

  def test_project_images(self):
    hyperparams = hyperparams_pb2.Hyperparams()
    text_format.Merge(
        r"""
        op: CONV
        activation: NONE
        regularizer {
          l2_regularizer {
            weight: 1e-8
          }
        }
        initializer {
          truncated_normal_initializer {
            mean: 0.0
            stddev: 0.03
          }
        }
        batch_norm {
          decay: 0.995
          center: true
          scale: false
          epsilon: 0.001
        }
        """, hyperparams)

    model_proto = gap_model_pb2.GAPModel()
    model = gap_model.Model(model_proto, is_training=False)

    # Project to 50-D feature space.

    tf.reset_default_graph()

    inputs = tf.random_uniform(shape=[32, 7, 7, 320])
    feature_map = model._project_images(
        inputs,
        common_dimensions=50,
        hyperparams=hyperparams,
        is_training=False)
    self.assertAllEqual(feature_map.get_shape().as_list(), [32, 7, 7, 50])

    # Project to 199-D feature space.

    tf.reset_default_graph()

    inputs = tf.random_uniform(shape=[32, 14, 14, 320])
    feature_map = model._project_images(
        inputs,
        common_dimensions=199,
        hyperparams=hyperparams,
        is_training=False)
    self.assertAllEqual(feature_map.get_shape().as_list(), [32, 14, 14, 199])

  def test_encode_captions(self):
    model_proto = gap_model_pb2.GAPModel()
    model = gap_model.Model(model_proto, is_training=False)

    # 50-D embedding, num_captions_in_batch=32, max_caption_length=20.

    tf.reset_default_graph()

    caption_strings = tf.fill([32, 20], "")
    caption_feature = model._encode_captions(
        caption_strings=caption_strings,
        vocabulary_list=["one", "two", "three", "four"],
        common_dimensions=50,
        scope="coco_word_embedding",
        is_training=False)

    with tf.variable_scope("input_layer/coco_word_embedding", reuse=True):
      word_embedding = tf.get_variable("embedding_weights", shape=[5, 50])

    self.assertAllEqual(word_embedding.get_shape().as_list(), [5, 50])
    self.assertAllEqual(caption_feature.get_shape().as_list(), [32, 20, 50])

    # 200-D embedding, num_captions_in_batch=32, max_caption_length=20.

    tf.reset_default_graph()

    caption_strings = tf.fill([32, 20], "")
    caption_feature = model._encode_captions(
        caption_strings=caption_strings,
        vocabulary_list=["one", "two", "three"],
        common_dimensions=200,
        scope="coco_word_embedding",
        is_training=False)

    with tf.variable_scope("input_layer/coco_word_embedding", reuse=True):
      word_embedding = tf.get_variable("embedding_weights", shape=[4, 200])

    self.assertAllEqual(word_embedding.get_shape().as_list(), [4, 200])
    self.assertAllEqual(caption_feature.get_shape().as_list(), [32, 20, 200])

  def test_calc_pairwise_similarity(self):
    model_proto = gap_model_pb2.GAPModel()
    model = gap_model.Model(model_proto, is_training=False)

    g = tf.Graph()

    with g.as_default():
      image_feature = tf.placeholder(tf.float32, shape=[None, None, None])
      text_feature = tf.placeholder(tf.float32, shape=[None, None, None])

      similarity = model._calc_pairwise_similarity(image_feature, text_feature)

      with self.test_session() as sess:

        # batch=2, num_regions=3, num_captions=2, max_caption_length=3.

        similarity_value = sess.run(
            similarity,
            feed_dict={
                image_feature: [[[1], [0], [0]], [[0], [2], [2]]],
                text_feature: [[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]]],
            })

        self.assertAllClose(similarity_value, [
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
             [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
             [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
             [[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]],
             [[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]]],
        ])


if __name__ == '__main__':
  tf.test.main()
