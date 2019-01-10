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
      input_pattern: "output/VOC2007_test_ssbox.record-00001-of-00020"
      interleave_cycle_length: 2
      is_training: true
      shuffle_buffer_size: 10
      batch_size: 7
      max_num_proposals: 2000
      image_resizer {
        fixed_shape_resizer {
          height: 448
          width: 448
        }
      }
    """
    self._options = reader_pb2.Reader()
    text_format.Merge(options_str, self._options)

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
          self.assertAllEqual(feature_values[InputDataFields.image].shape,
                              [7, 448, 448, 3])
          self.assertAllEqual(feature_values[InputDataFields.image_id].shape,
                              [7])
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
