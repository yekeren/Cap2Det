from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf
import json
import nltk.tokenize

tf.flags.DEFINE_string('train_caption_annotations_file', '',
                       'Training annotations JSON file.')

tf.flags.DEFINE_string('vocabulary_file', '', 'Vocabulary words used in model.')

FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

_INIT_WIDTH = 0.03


def _process_caption(caption):
  """Processes a caption string into a list of tonenized words.

  Args:
    caption: A string caption.
  Returns:
    A list of strings; the tokenized caption.
  """
  return nltk.tokenize.word_tokenize(caption.lower())


def _load_vocabulary(fid):
  return set([x.split('\t')[0] for x in fid.readlines()])


def main(_):
  with tf.gfile.GFile(FLAGS.train_caption_annotations_file, 'r') as cap_fid:
    caption_groundtruth_data = json.load(cap_fid)

  assert 'annotations' in caption_groundtruth_data

  with tf.gfile.GFile(FLAGS.vocabulary_file, 'r') as voc_fid:
    vocabulary_data = _load_vocabulary(voc_fid)

  tf.logging.info("voc: %s", "\n".join(vocabulary_data))
  tf.logging.info("voc size: %d", len(vocabulary_data))

  # Compute word frequency and filter out rare words.

  covered, uncovered = 0, 0

  caption_words = set([])
  for annotation in caption_groundtruth_data['annotations']:
    for word in _process_caption(annotation['caption']):
      if word in vocabulary_data:
        covered += 1
        break
    else:
      uncovered += 1

  tf.logging.info("-" * 50)
  tf.logging.info("Total: %d", covered + uncovered)
  tf.logging.info("Covered: %d", covered)
  tf.logging.info("Uncovered: %d", uncovered)
  tf.logging.info("-" * 50)


if __name__ == '__main__':
  tf.app.run()
