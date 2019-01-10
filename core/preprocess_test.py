from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from google.protobuf import text_format

import os
import cv2
from core import preprocess
from protos import preprocess_pb2

_TESTDATA = "testdata"
_TESTFILE = "114144.jpg"
_TMPDIR = "tmp"

tf.logging.set_verbosity(tf.logging.INFO)


class PreprocessTest(tf.test.TestCase):

  def test_random_crop(self):
    tf.reset_default_graph()

    scale = tf.placeholder(tf.float32, [])
    image = tf.placeholder(tf.uint8, [None, None, 3])

    cropped = preprocess.random_crop(image, random_crop_min_scale=scale)

    filename = os.path.join(_TESTDATA, _TESTFILE)
    image_data = cv2.imread(filename)[:, :, ::-1].copy()

    with self.test_session() as sess:
      for scale_data in [1.0, 0.8, 0.6, 0.4, 0.2]:
        roi = sess.run(
            cropped, feed_dict={
                image: image_data,
                scale: scale_data
            })
        filename = os.path.join(_TMPDIR, "%.1lf-%s" % (scale_data, _TESTFILE))
        cv2.imwrite(filename, roi[:, :, ::-1])
        tf.logging.info("ROI image at scale %.1lf is written to %s.",
                        scale_data, filename)

  def _preprocess(self, options, prefix):
    g = tf.Graph()
    with g.as_default():
      image = tf.placeholder(tf.uint8, [None, None, 3])
      preprocessed = preprocess.preprocess_image(image, options)

      filename = os.path.join(_TESTDATA, _TESTFILE)
      image_data = cv2.imread(filename)[:, :, ::-1].copy()

    with self.test_session(graph=g) as sess:
      result = sess.run(preprocessed, feed_dict={image: image_data})
      filename = os.path.join(_TMPDIR, "%s-%s" % (prefix, _TESTFILE))
      cv2.imwrite(filename, result[:, :, ::-1])
      tf.logging.info("Preprocessed image is written to %s.", filename)

  def test_flip(self):
    options = preprocess_pb2.Preprocess()
    text_format.Merge(r"""
        random_flip_left_right_prob: 0.5
        """, options)
    self._preprocess(options, "flip_no")

    options.random_flip_left_right_prob = 1.0
    self._preprocess(options, "flip_yes")

  def test_brightness(self):
    for max_delta in [0.4, 0.8]:
      options = preprocess_pb2.Preprocess()
      text_format.Merge(
          r"""
          random_brightness_prob: 1.0
          random_brightness_max_delta: %.2lf
          """ % (max_delta), options)
      self._preprocess(options, "brightness_%.2lf" % (max_delta))

  def test_contrast(self):
    for contrast in [0.4, 0.6, 0.8]:
      options = preprocess_pb2.Preprocess()
      text_format.Merge(
          r"""
          random_contrast_prob: 1.0
          random_contrast_lower: %.2lf
          random_contrast_upper: %.2lf
          """ % (contrast, contrast + 0.01), options)
      self._preprocess(options, "contrast_%.2lf" % (contrast))

  def test_hue(self):
    for max_delta in [0.05, 0.10, 0.15]:
      options = preprocess_pb2.Preprocess()
      text_format.Merge(
          r"""
          random_hue_prob: 1.0
          random_hue_max_delta: %.2lf
          """ % (max_delta), options)
      self._preprocess(options, "hue_%.2lf" % (max_delta))

  def test_saturation(self):
    for saturation in [0.4, 1.6]:
      options = preprocess_pb2.Preprocess()
      text_format.Merge(
          r"""
          random_saturation_prob: 1.0
          random_saturation_lower: %.2lf
          random_saturation_upper: %.2lf
          """ % (saturation, saturation + 0.01), options)
      self._preprocess(options, "saturation_%.2lf" % (saturation))

  def test_preprocess(self):
    for index in range(20):
      options = preprocess_pb2.Preprocess()
      text_format.Merge(
          r"""
          random_flip_left_right_prob: 0.5
          random_crop_prob: 1.0
          random_crop_min_scale: 0.6
          random_brightness_prob: 0.8
          random_brightness_max_delta: 0.2
          random_contrast_prob: 0.8
          random_contrast_lower: 0.7
          random_contrast_upper: 1.0
          random_hue_prob: 0.2
          random_hue_max_delta: 0.10
          random_saturation_prob: 0.8
          random_saturation_lower: 0.6
          random_saturation_upper: 1.4
          """, options)
      self._preprocess(options, "preprocess_%i" % (index))

  def test_parse_texts(self):
    tf.reset_default_graph()

    tokens = tf.placeholder(dtype=tf.string, shape=[None])
    offsets = tf.placeholder(dtype=tf.int64, shape=[None])
    lengths = tf.placeholder(dtype=tf.int64, shape=[None])

    # Lengths of offsets and lengths are not matched.

    (num_texts, text_strings, text_lengths) = preprocess.parse_texts(
        tokens, offsets, lengths)

    with self.test_session() as sess:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        (num_caps, cap_strings, cap_lengths) = sess.run(
            [num_texts, text_strings, text_lengths],
            feed_dict={
                tokens: ["first", "second", "text", "the", "third", "text"],
                offsets: [0, 1],
                lengths: [1, 2, 3]
            })

    # Basic tests.

    (num_texts, text_strings, text_lengths) = preprocess.parse_texts(
        tokens, offsets, lengths)
    with self.test_session() as sess:
      (num_caps, cap_strings, cap_lengths) = sess.run(
          [num_texts, text_strings, text_lengths],
          feed_dict={
              tokens: ["first", "second", "text", "the", "third", "text"],
              offsets: [0, 1, 3],
              lengths: [1, 2, 3]
          })
      self.assertEqual(num_caps, 3)
      self.assertAllEqual(cap_strings,
                          [[b"first", b"", b""], [b"second", b"text", b""],
                           [b"the", b"third", b"text"]])
      self.assertAllEqual(cap_lengths, [1, 2, 3])


if __name__ == '__main__':
  tf.test.main()
