r"""Generate the vocabulary file for the Flickr30K dataset.

Example usage:
    python create_flickr30k_vocab.py \
      --logtostderr \
      --annotation_path="${ANNOTATION_PATH}" \
      --glove_file="${GLOVE_FILE}" \
      --output_vocabulary_file="${OUTPUT_VOCABULARY_FILE}"
      --output_vocabulary_word_embedding_file="${OUTPUT_VOCABULARY_WORD_EMBEDDING_FILE}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk.tokenize
import numpy as np

import tensorflow as tf
import collections

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags

flags.DEFINE_string('annotation_path', '', '')

tf.flags.DEFINE_string('glove_file', '',
                       'Path to the pre-trained GloVe embedding file.')

tf.flags.DEFINE_string('output_vocabulary_file', '',
                       'Vocabulary file to export.')

tf.flags.DEFINE_string('output_vocabulary_word_embedding_file', '',
                       'Vocabulary weights file to export.')

tf.flags.DEFINE_integer('min_word_freq', 10, 'Minimum word frequency.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


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


def _process_caption(caption):
  """Processes a caption string into a list of tonenized words.

  Args:
    caption: A string caption.
  Returns:
    A list of strings; the tokenized caption.
  """
  return nltk.tokenize.word_tokenize(caption.lower())


def _load_annotations(filepath):
  annotations = {}
  with open(filepath, 'r', encoding='utf-8') as fid:
    for line in fid.readlines():
      name_str, caption = line.strip('\n').split('\t')
      image_id = int(name_str.split('.')[0])
      annotation = annotations.setdefault(image_id, [])
      annotation.append(caption)
  return annotations


def main(_):
  glove = _load_glove(FLAGS.glove_file)
  annotations = _load_annotations(FLAGS.annotation_path)

  word_freq = collections.Counter()
  for image_id, annotation in annotations.items():
    for caption in annotation:
      for word in _process_caption(caption):
        word_freq[word] += 1

  word_freq = [
      x for x in word_freq.most_common()
      if x[1] >= FLAGS.min_word_freq and x[0] in glove
  ]
  with tf.gfile.GFile(FLAGS.output_vocabulary_file, 'w') as fp:
    for word, freq in word_freq:
      fp.write('%s\n' % (word))

  # Generate initial word embeddings.

  unk, embeddings = 0, []
  for word, freq in word_freq:
    assert word in glove
    embeddings.append(glove[word])
  embeddings = np.stack(embeddings, axis=0)

  with tf.gfile.GFile(FLAGS.output_vocabulary_word_embedding_file, 'wb') as fp:
    np.save(fp, embeddings)
  tf.logging.info("Shape of word embeddings: %s", embeddings.shape)
  tf.logging.info('UNK words: %s', unk)

  tf.logging.info('Done')


if __name__ == '__main__':
  tf.app.run()
