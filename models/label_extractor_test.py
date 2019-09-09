from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from google.protobuf import text_format
from core.standard_fields import InputDataFields

from protos import label_extractor_pb2
from models import label_extractor


class LabelExtractorTest(tf.test.TestCase):

  def test_groundtruth_extractor(self):
    filename = '/tmp/label_file.txt'
    with tf.gfile.Open(filename, 'w') as f:
      f.write('\n'.join(['person', 'bird', 'dining table']))

    proto_str = r"""
      groundtruth_extractor {
        label_file: '%s'
      }
    """ % filename
    options = label_extractor_pb2.LabelExtractor()
    text_format.Merge(proto_str, options)

    extractor = label_extractor.build_label_extractor(options)
    self.assertIsInstance(extractor, label_extractor.GroundtruthExtractor)
    self.assertEqual(extractor.num_classes, 3)
    self.assertListEqual(extractor.classes, ['person', 'bird', 'dining table'])

    object_texts = tf.placeholder(dtype=tf.string, shape=[4, None])
    labels = extractor.extract_labels(
        examples={InputDataFields.object_texts: object_texts})
    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      self.assertAllEqual([[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 0]],
                          sess.run(
                              labels,
                              feed_dict={
                                  object_texts:
                                  [['bird', 'person', 'dining table'],
                                   ['dining table', '', ''],
                                   ['bird', 'dining table', ''],
                                   ['class_?', 'class_*', 'class_%']]
                              }))

  def test_exact_match_extractor(self):
    filename = '/tmp/label_file.txt'
    with tf.gfile.Open(filename, 'w') as f:
      f.write('\n'.join(['person', 'bird', 'dining table']))

    proto_str = r"""
      exact_match_extractor {
        label_file: '%s'
      }
    """ % filename
    options = label_extractor_pb2.LabelExtractor()
    text_format.Merge(proto_str, options)

    extractor = label_extractor.build_label_extractor(options)
    self.assertIsInstance(extractor, label_extractor.ExactMatchExtractor)
    self.assertEqual(extractor.num_classes, 3)
    self.assertListEqual(extractor.classes, ['person', 'bird', 'dining table'])

    object_texts = tf.placeholder(dtype=tf.string, shape=[4, None])
    labels = extractor.extract_labels(
        examples={InputDataFields.concat_caption_string: object_texts})
    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      self.assertAllEqual(
          [[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 0]],
          sess.run(
              labels,
              feed_dict={
                  object_texts: [['bird', 'person', 'table'], ['table', '', ''],
                                 ['bird', 'table', ''],
                                 ['class_?', 'class_*', 'class_%']]
              }))

  def test_extend_match_extractor(self):
    filename = '/tmp/label_file.txt'
    with tf.gfile.Open(filename, 'w') as f:
      f.write('\n'.join([
          'person\tgirl,boy,man,child,adult,rider',
          'bird\tgoose,duck,pelican,flamigo,gull,swan,bluejay',
          'dining table\ttable', 'tie\t'
      ]))

    proto_str = r"""
      extend_match_extractor {
        label_file: '%s'
      }
    """ % filename
    options = label_extractor_pb2.LabelExtractor()
    text_format.Merge(proto_str, options)

    extractor = label_extractor.build_label_extractor(options)
    self.assertIsInstance(extractor, label_extractor.ExtendMatchExtractor)
    self.assertEqual(extractor.num_classes, 4)
    self.assertListEqual(extractor.classes,
                         ['person', 'bird', 'dining table', 'tie'])

    object_texts = tf.placeholder(dtype=tf.string, shape=[4, None])
    labels = extractor.extract_labels(
        examples={InputDataFields.concat_caption_string: object_texts})
    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      self.assertAllEqual(
          [[1, 1, 1, 0], [0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]],
          sess.run(
              labels,
              feed_dict={
                  object_texts: [['goose', 'boy', 'table'], ['table', '', ''],
                                 ['swan', 'girl', ''],
                                 ['class_?', 'class_*', 'tie']]
              }))


if __name__ == '__main__':
  tf.test.main()
