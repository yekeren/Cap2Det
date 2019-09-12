r"""Generate the vocabulary file for the MSCOCO dataset.

Example usage:
  python "tools/create_coco_vocab.py" \
    --logtostderr \
    --train_caption_annotations_file="${TRAIN_CAPTION_ANNOTATIONS_FILE}" \
    --glove_file="${GLOVE_FILE}" \
    --output_vocabulary_file="${OUTPUT_VOCABULARY_FILE}" \
    --output_vocabulary_word_embedding_file="${OUTPUT_VOCABULARY_WORD_EMBEDDING_FILE}" \
    --min_word_freq=${MIN_WORD_FREQ}
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import collections
import nltk.tokenize

import numpy as np
import tensorflow as tf

tf.flags.DEFINE_string('train_caption_annotations_file', '',
                       'Training annotations JSON file.')

tf.flags.DEFINE_string('glove_file', '',
                       'Path to the pre-trained GloVe embedding file.')

tf.flags.DEFINE_string('output_vocabulary_file', '',
                       'Vocabulary file to be exported.')

tf.flags.DEFINE_string('output_vocabulary_word_embedding_file', '',
                       'Vocabulary word embedding file to be exported.')

tf.flags.DEFINE_integer('min_word_freq', 10, 'Minimum word frequency.')

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
    word2vec[word] = np.array(vec)
    if i % 10000 == 0:
      tf.logging.info('On load %s/%s', i, len(lines))
  return word2vec


def main(_):
  glove = _load_glove(FLAGS.glove_file)

  with tf.gfile.GFile(FLAGS.train_caption_annotations_file, 'r') as f:
    caption_groundtruth_data = json.load(f)

  # Compute word frequency and filter out rare words.

  word_freq = collections.Counter()
  for annotation in caption_groundtruth_data['annotations']:
    for word in _process_caption(annotation['caption']):
      word_freq[word] += 1
  tf.logging.info('Processed %i captions.',
                  len(caption_groundtruth_data['annotations']))

  word_freq = [
      x for x in word_freq.most_common()
      if x[1] >= FLAGS.min_word_freq and x[0] in glove
  ]

  # Output the vocabulary file.

  with tf.gfile.GFile(FLAGS.output_vocabulary_file, 'w') as f:
    for word, freq in word_freq:
      f.write('%s\n' % word)

  # Output the word embedding file.

  embeddings = []
  for word, freq in word_freq:
    embeddings.append(glove[word])
  embeddings = np.stack(embeddings, axis=0)

  with tf.gfile.GFile(FLAGS.output_vocabulary_word_embedding_file, 'wb') as fp:
    np.save(fp, embeddings)
  tf.logging.info("Shape of word embeddings: %s", embeddings.shape)


if __name__ == '__main__':
  tf.app.run()
