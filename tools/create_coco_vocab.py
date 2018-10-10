
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

tf.flags.DEFINE_string('glove_file', '',
                       'Path to the pre-trained GloVe embedding file.')

tf.flags.DEFINE_string('vocabulary_file', '',
                       'Vocabulary file to export.')

tf.flags.DEFINE_string('vocabulary_weights_file', '',
                       'Vocabulary weights file to export.')

tf.flags.DEFINE_integer('min_word_freq', 20, 
                        'Minimum word frequency.')

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


def _load_glove(filename):
  """Loads the pre-trained GloVe word embedding.

  Args:
    filename: path to the GloVe embedding file.

  Returns:
    word embedding dict keyed by word.
  """
  with tf.gfile.GFile(filename, 'r') as fid:
    lines = fid.readlines()

  num_words = len(lines)
  embedding_size = len(lines[0].strip('\n').split()) - 1
  word2vec = {}
  for i, line in enumerate(lines):
    items = line.strip('\n').split()
    word, vec = items[0], [float(v) for v in items[1:]]
    assert len(vec) == embedding_size
    word2vec[word] = np.array(vec)
    if i % 10000== 0:
      tf.logging.info('On load %s/%s', i, len(lines))
  return word2vec


def main(_):
  with tf.gfile.GFile(FLAGS.train_caption_annotations_file, 'r') as cap_fid:
    caption_groundtruth_data = json.load(cap_fid)

  assert 'annotations' in caption_groundtruth_data

  # Compute word frequency and filter out rare words.

  word_freq = collections.Counter()
  for annotation in caption_groundtruth_data['annotations']:
    for word in _process_caption(annotation['caption']):
      word_freq[word] += 1

  word_freq = [x for x in word_freq.most_common() if x[1] >= FLAGS.min_word_freq]
  with tf.gfile.GFile(FLAGS.vocabulary_file, 'w') as fp:
    for word, freq in word_freq:
      fp.write('%s\n' % (word))

  # Generate initial word embeddings.

  glove = _load_glove(FLAGS.glove_file)
  dims = glove['the'].shape[0]

  unk, embeddings = 0, []
  for word, freq in word_freq:
    if word in glove:
      embeddings.append(glove[word])
    else:
      unk += 1
      tf.logging.warning('Unknown word %s.', word)
      embeddings.append(_INIT_WIDTH * (np.random.rand(dims) * 2 - 1))
  embeddings = np.stack(embeddings, axis=0)

  with tf.gfile.GFile(FLAGS.vocabulary_weights_file, 'wb') as fp:
    np.save(fp, embeddings)
  tf.logging.info("Shape of word embeddings: %s", embeddings.shape)
  tf.logging.info('UNK words: %s', unk)

  tf.logging.info('Done')


if __name__ == '__main__':
  tf.app.run()
