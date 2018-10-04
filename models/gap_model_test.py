
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from models import gap_model
from protos import gap_model_pb2

tf.logging.set_verbosity(tf.logging.INFO)


class GAPModelTest(tf.test.TestCase):

  def test_encode_images(self):
    model_proto = gap_model_pb2.GAPModel()
    model = gap_model.Model(model_proto, is_training=False)

    # MobilenetV2, project to 300 channels.

    tf.reset_default_graph()

    image = tf.random_uniform(shape=[32, 224, 224, 3])
    feature_map = model._encode_images(image, 
        cnn_name="mobilenet_v2", 
        cnn_weight_decay=1e-4,
        cnn_feature_map="layer_18/output",
        cnn_checkpoint=None,
        common_dimensions=300,
        scope="image_proj",
        is_training=False)
    self.assertAllEqual(feature_map.get_shape().as_list(), [32, 7, 7, 300])

    # MobilenetV2, project to 50 channels.

    tf.reset_default_graph()

    image = tf.random_uniform(shape=[32, 448, 448, 3])
    feature_map = model._encode_images(image, 
        cnn_name="mobilenet_v2", 
        cnn_weight_decay=1e-4,
        cnn_feature_map="layer_18/output",
        cnn_checkpoint=None,
        common_dimensions=50,
        scope="image_proj",
        is_training=False)
    self.assertAllEqual(feature_map.get_shape().as_list(), [32, 14, 14, 50])

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

        similarity_value = sess.run(similarity, feed_dict={
            image_feature: [[[1], [0], [0]], [[0], [2], [2]]],
            text_feature: [[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]]],
            })

        self.assertAllClose(similarity_value, 
            [[
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], 
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ], [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 
            [[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]], 
            [[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]]],
            ])

        # # batch=2, num_regions=3, num_captions=2, max_caption_length=3,
        # # common_dimensions=1, caption_lenths=[1, 3].

        # similarity_value = sess.run(similarity, feed_dict={
        #     image_feature: [[[1], [0], [0]], [[0], [2], [2]]],
        #     text_feature: [[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]]],
        #     caption_lengths: [1, 3]})

        # self.assertAllClose(similarity_value, [
        #     [(0.1) / 3.0, (0.4 + 0.5 + 0.6) / 9.0],
        #     [(0.1) * 4 / 3.0, (0.4 + 0.5 + 0.6) * 4 / 9.0]])

        # # batch=2, num_regions=3, num_captions=2, max_caption_length=3,
        # # common_dimensions=1, caption_lenths=[3, 1].

        # similarity_value = sess.run(similarity, feed_dict={
        #     image_feature: [[[1], [0], [0]], [[0], [2], [2]]],
        #     text_feature: [[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]]],
        #     caption_lengths: [3, 1]})

        # self.assertAllClose(similarity_value, [
        #     [(0.1 + 0.2 + 0.3) / 9.0, (0.4) / 3.0],
        #     [(0.1 + 0.2 + 0.3) * 4 / 9.0, (0.4) * 4 / 3.0]])



if __name__ == '__main__':
  tf.test.main()
