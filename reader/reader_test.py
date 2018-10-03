
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from google.protobuf import text_format

import reader

from core.standard_fields import InputDataFields
from core.standard_fields import TFExampleDataFields
from protos import reader_pb2

tf.logging.set_verbosity(tf.logging.INFO)


class ReaderTest(tf.test.TestCase):
  def setUp(self):
    options_str = r"""
      input_pattern: "output/coco_train.record-00000-of-00100" 
      interleave_cycle_length: 2
      is_training: true
      shuffle_buffer_size: 1000
      batch_size: 7
      image_height: 224
      image_width: 224
      image_channels: 3
      max_caption_length: 19
    """
    self._options = reader_pb2.Reader()
    text_format.Merge(options_str, self._options)

  def test_parse_captions(self):
    tf.reset_default_graph()

    tokens = tf.placeholder(dtype=tf.string, shape=[None])
    offsets = tf.placeholder(dtype=tf.int64, shape=[None])
    lengths = tf.placeholder(dtype=tf.int64, shape=[None])

    # Lengths of offsets and lengths are not matched.

    (num_captions, caption_strings, caption_lengths
     ) = reader.parse_captions(tokens, offsets, lengths)

    with self.test_session() as sess:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        (num_caps, cap_strings, cap_lengths) = sess.run(
            [num_captions, caption_strings, caption_lengths], feed_dict={ 
              tokens: ["first", "second", "caption", "the", "third", "caption"],
              offsets: [0, 1],
              lengths: [1, 2, 3]})

    # Basic, max_caption_length=4.

    (num_captions, caption_strings, caption_lengths
     ) = reader.parse_captions(tokens, offsets, lengths, max_caption_length=4)
    with self.test_session() as sess:
      (num_caps, cap_strings, cap_lengths) = sess.run(
          [num_captions, caption_strings, caption_lengths], feed_dict={ 
            tokens: ["first", "second", "caption", "the", "third", "caption"],
            offsets: [0, 1, 3],
            lengths: [1, 2, 3]})
      self.assertEqual(num_caps, 3)
      self.assertAllEqual(cap_strings, [
          [b"first", b"", b"", b""], 
          [b"second", b"caption", b"", b""], 
          [b"the", b"third", b"caption", b""]])
      self.assertAllEqual(cap_lengths, [1, 2, 3])

    # Trim, max_caption_length=2, also need to modify cap_lengths.

    (num_captions, caption_strings, caption_lengths
     ) = reader.parse_captions(tokens, offsets, lengths, max_caption_length=2)
    with self.test_session() as sess:
      (num_caps, cap_strings, cap_lengths) = sess.run(
          [num_captions, caption_strings, caption_lengths], feed_dict={ 
            tokens: ["first", "second", "caption", "the", "third", "caption"],
            offsets: [0, 1, 3],
            lengths: [1, 2, 3]})
      self.assertEqual(num_caps, 3)
      self.assertAllEqual(cap_strings, [
          [b"first", b""], 
          [b"second", b"caption"], 
          [b"the", b"third"]])
      self.assertAllEqual(cap_lengths, [1, 2, 2])

  def test_get_examples(self):
    tf.reset_default_graph()

    input_fn = reader.get_input_fn(self._options)

    dataset = input_fn()
    iterator = dataset.make_initializable_iterator()
    feature_dict = iterator.get_next()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      try:
        i = 0
        while not coord.should_stop():
          feature_values = sess.run(feature_dict)
          self.assertSetEqual(
              set(feature_values.keys()),
              set([
                InputDataFields.image, 
                InputDataFields.image_id,
                InputDataFields.num_captions,
                InputDataFields.caption_strings,
                InputDataFields.caption_lengths]))
          self.assertAllEqual(
              feature_values[InputDataFields.image].shape, [7, 224, 224, 3])
          self.assertAllEqual(
              feature_values[InputDataFields.image_id].shape, [7])
          self.assertAllEqual(
              feature_values[InputDataFields.num_captions].shape, [7])

          self.assertEqual(
              len(feature_values[InputDataFields.caption_lengths].shape), 2)
          self.assertEqual(
              len(feature_values[InputDataFields.caption_strings].shape), 3)

          self.assertEqual(
              feature_values[InputDataFields.caption_lengths].shape[0], 7)
          self.assertEqual(
              feature_values[InputDataFields.caption_strings].shape[0], 7)
          self.assertEqual(
              feature_values[InputDataFields.caption_strings].shape[1], 
              feature_values[InputDataFields.caption_lengths].shape[1])
          self.assertEqual(
              feature_values[InputDataFields.caption_strings].shape[2], 19)
          i += 1
          if i >= 10:
            coord.request_stop()
      except tf.errors.OutOfRangeError:
        tf.logging.info("OutOfRangeError, done!")
      finally:
        tf.logging.info("Request to stop!")
        coord.request_stop()
      coord.join(threads)


if __name__ == '__main__':
  tf.test.main()
