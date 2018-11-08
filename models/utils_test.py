from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models import utils


class UtilsTest(tf.test.TestCase):

  def test_gather_in_batch_captions(self):
    g = tf.Graph()

    with g.as_default():
      image_id = tf.placeholder(tf.string, [None])
      num_captions = tf.placeholder(tf.int32, [None])
      caption_strings = tf.placeholder(tf.string, [None, None, None])
      caption_lengths = tf.placeholder(tf.int32, [None, None])

      (image_ids_gathered, caption_strings_gathered,
       caption_lengths_gathered) = utils.gather_in_batch_captions(
           image_id, num_captions, caption_strings, caption_lengths)

    with self.test_session(graph=g) as sess:

      # batch=2, max_num_captions=3, max_caption_length=3, num_captions=[3, 1].

      (image_ids_value, caption_strings_value,
       caption_lengths_value) = sess.run(
           [
               image_ids_gathered, caption_strings_gathered,
               caption_lengths_gathered
           ],
           feed_dict={
               image_id: ["a", "b"],
               num_captions: [3, 1],
               caption_strings: [[["a", "a", "a"], ["b", "b", "b"],
                                  ["c", "c", "c"]],
                                 [["d", "d", "d"], ["e", "e", "e"],
                                  ["f", "f", "f"]]],
               caption_lengths: [[1, 2, 3], [2, 3, 3]],
           })
      self.assertAllEqual(image_ids_value, [b"a", b"a", b"a", b"b"])
      self.assertAllEqual(caption_strings_value,
                          [[b"a", b"a", b"a"], [b"b", b"b", b"b"],
                           [b"c", b"c", b"c"], [b"d", b"d", b"d"]])
      self.assertAllEqual(caption_lengths_value, [1, 2, 3, 2])

      # batch=2, max_num_captions=3, max_caption_length=3, num_captions=[2, 2].

      (image_ids_value, caption_strings_value,
       caption_lengths_value) = sess.run(
           [
               image_ids_gathered, caption_strings_gathered,
               caption_lengths_gathered
           ],
           feed_dict={
               image_id: ["a", "b"],
               num_captions: [2, 2],
               caption_strings: [[["a", "a", "a"], ["b", "b", "b"],
                                  ["c", "c", "c"]],
                                 [["d", "d", "d"], ["e", "e", "e"],
                                  ["f", "f", "f"]]],
               caption_lengths: [[1, 2, 3], [2, 3, 3]],
           })
      self.assertAllEqual(image_ids_value, [b"a", b"a", b"b", b"b"])
      self.assertAllEqual(caption_strings_value,
                          [[b"a", b"a", b"a"], [b"b", b"b", b"b"],
                           [b"d", b"d", b"d"], [b"e", b"e", b"e"]])
      self.assertAllEqual(caption_lengths_value, [1, 2, 2, 3])

      # batch=2, max_num_captions=3, max_caption_length=3, num_captions=[1, 3].

      (image_ids_value, caption_strings_value,
       caption_lengths_value) = sess.run(
           [
               image_ids_gathered, caption_strings_gathered,
               caption_lengths_gathered
           ],
           feed_dict={
               image_id: ["a", "b"],
               num_captions: [1, 3],
               caption_strings: [[["a", "a", "a"], ["b", "b", "b"],
                                  ["c", "c", "c"]],
                                 [["d", "d", "d"], ["e", "e", "e"],
                                  ["f", "f", "f"]]],
               caption_lengths: [[1, 2, 3], [2, 3, 3]],
           })
      self.assertAllEqual(image_ids_value, [b"a", b"b", b"b", b"b"])
      self.assertAllEqual(caption_strings_value,
                          [[b"a", b"a", b"a"], [b"d", b"d", b"d"],
                           [b"e", b"e", b"e"], [b"f", b"f", b"f"]])
      self.assertAllEqual(caption_lengths_value, [1, 2, 3, 3])


if __name__ == '__main__':
  tf.test.main()
