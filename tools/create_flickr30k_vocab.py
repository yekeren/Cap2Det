# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert raw COCO dataset to TFRecord for object_detection.

Please note that this tool creates sharded output files.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_file="${TRAIN_IMAGE_FILE}" \
      --val_image_file="${VAL_IMAGE_FILE}" \
      --test_image_file="${TEST_IMAGE_FILE}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --train_caption_annotations_file="${TRAIN_CAPTION_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --val_caption_annotations_file="${VAL_CAPTION_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import hashlib
import io
import json
import os
import contextlib2
import nltk.tokenize
import numpy as np
import PIL.Image
import zipfile

import tensorflow as tf
import collections

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags

flags.DEFINE_string('annotation_path', '', '')
flags.DEFINE_string('output_path', '', '')
tf.flags.DEFINE_string('glove_file', '',
                       'Path to the pre-trained GloVe embedding file.')

tf.flags.DEFINE_string('vocabulary_file', '', 'Vocabulary file to export.')

tf.flags.DEFINE_string('category_file', '', 'Category file to load.')

tf.flags.DEFINE_string('vocabulary_weights_file', '',
                       'Vocabulary weights file to export.')

tf.flags.DEFINE_integer('min_word_freq', 20, 'Minimum word frequency.')

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
  annotations = _load_annotations(FLAGS.annotation_path)
  glove = _load_glove(FLAGS.glove_file)

  mapping = {
      'aeroplane': 'airplane',
      'diningtable': 'table',
      'pottedplant': 'plant',
      'tvmonitor': 'tv',
  }

  with open(FLAGS.category_file, 'r') as fid:
    categories = set([x.strip('\n') for x in fid.readlines()])
    categories = set([mapping.get(x, x) for x in categories])

  total = 0
  total_bingo = 0
  word_freq = collections.Counter()
  for image_id, annotation in annotations.items():
    for caption in annotation:
      bingo = 0
      for word in _process_caption(caption):
        word_freq[word] += 1
        if word in categories:
          bingo = 1
      total_bingo += bingo
      total += 1

  tf.logging.info('Recalled %i / %i captions.', total_bingo, total)

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
