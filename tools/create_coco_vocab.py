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

tf.flags.DEFINE_string('vocabulary_file', '', 'Vocabulary file to export.')

tf.flags.DEFINE_string('category_file', '', 'Category file to load.')

tf.flags.DEFINE_string('vocabulary_weights_file', '',
                       'Vocabulary weights file to export.')

tf.flags.DEFINE_integer('min_word_freq', 20, 'Minimum word frequency.')

FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


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
    if i % 10000 == 0:
      tf.logging.info('On load %s/%s', i, len(lines))
  return word2vec


def main(_):
  with tf.gfile.GFile(FLAGS.train_caption_annotations_file, 'r') as cap_fid:
    caption_groundtruth_data = json.load(cap_fid)
  assert 'annotations' in caption_groundtruth_data

  mapping = {
      'traffic light': 'stoplight',
      'fire hydrant': 'hydrant',
      'stop sign': 'sign',
      'parking meter': 'meter',
      'sports ball': 'ball',
      'baseball bat': 'bat',
      'baseball glove': 'glove',
      'tennis racket': 'racket',
      'wine glass': 'wineglass',
      'hot dog': 'hotdog',
      'potted plant': 'plant',
      'dining table': 'table',
      'cell phone': 'cellphone',
      'teddy bear': 'teddy',
      'hair drier': 'hairdryer',
  }
  with open(FLAGS.category_file, 'r') as fid:
    categories = [x.strip('\n') for x in fid.readlines()]
    categories = set([mapping.get(x, x) for x in categories])

  glove = _load_glove(FLAGS.glove_file)

  # Compute word frequency and filter out rare words.

  total_bingo = 0
  word_freq = collections.Counter()
  for annotation in caption_groundtruth_data['annotations']:

    bingo = 0
    for word in _process_caption(annotation['caption']):
      word_freq[word] += 1
      if word in categories:
        bingo = 1
    total_bingo += bingo
  tf.logging.info('Recalled %i / %i captions.', total_bingo,
                  len(caption_groundtruth_data['annotations']))

  word_freq = [
      x for x in word_freq.most_common()
      if x[1] >= FLAGS.min_word_freq and x[0] in glove
  ]
  with tf.gfile.GFile(FLAGS.vocabulary_file, 'w') as fp:
    for word, freq in word_freq:
      fp.write('%s\n' % (word))

  # Generate initial word embeddings.

  unk, embeddings = 0, []
  for word, freq in word_freq:
    assert word in glove
    embeddings.append(glove[word])
  embeddings = np.stack(embeddings, axis=0)

  with tf.gfile.GFile(FLAGS.vocabulary_weights_file, 'wb') as fp:
    np.save(fp, embeddings)
  tf.logging.info("Shape of word embeddings: %s", embeddings.shape)
  tf.logging.info('UNK words: %s', unk)

  tf.logging.info('Done')


if __name__ == '__main__':
  tf.app.run()
