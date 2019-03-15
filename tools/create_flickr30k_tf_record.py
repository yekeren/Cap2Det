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

flags.DEFINE_string('image_tar_file', '', '')
flags.DEFINE_string('proposal_data', '', '')
flags.DEFINE_string('annotation_path', '', '')
flags.DEFINE_string('output_path', '', '')
flags.DEFINE_integer('number_of_parts', 20, 'Number of output parts.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

counter = collections.defaultdict(int)


def _process_caption(caption):
  """Processes a caption string into a list of tonenized words.

  Args:
    caption: A string caption.
  Returns:
    A list of strings; the tokenized caption.
  """
  return nltk.tokenize.word_tokenize(caption.lower())


def _create_tf_example(image_id, annotation, encoded_jpg):
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  height, width = image.height, image.width
  key = hashlib.sha256(encoded_jpg).hexdigest()

  npy_path = os.path.join(FLAGS.proposal_data, '{}.npy'.format(image_id))

  with open(npy_path, 'rb') as fid:
    proposals = np.load(fid)

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []

  caption_string = []
  caption_offset = []
  caption_length = []
  for caption in annotation:
    caption = _process_caption(caption)
    caption_offset.append(len(caption_string))
    caption_length.append(len(caption))
    caption_string.extend(caption)
  caption_string = [caption.encode('utf8') for caption in caption_string]

  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/height':
              dataset_util.int64_feature(height),
              'image/width':
              dataset_util.int64_feature(width),
              'image/filename':
              dataset_util.bytes_feature(str(image_id).encode('utf8')),
              'image/source_id':
              dataset_util.bytes_feature(str(image_id).encode('utf8')),
              'image/key/sha256':
              dataset_util.bytes_feature(key.encode('utf8')),
              'image/encoded':
              dataset_util.bytes_feature(encoded_jpg),
              'image/format':
              dataset_util.bytes_feature('jpeg'.encode('utf8')),
              'image/object/bbox/xmin':
              dataset_util.float_list_feature(xmin),
              'image/object/bbox/xmax':
              dataset_util.float_list_feature(xmax),
              'image/object/bbox/ymin':
              dataset_util.float_list_feature(ymin),
              'image/object/bbox/ymax':
              dataset_util.float_list_feature(ymax),
              'image/object/class/text':
              dataset_util.bytes_list_feature(classes_text),
              'image/object/class/label':
              dataset_util.int64_list_feature(classes),
              'image/object/difficult':
              dataset_util.int64_list_feature(difficult_obj),
              'image/object/truncated':
              dataset_util.int64_list_feature(truncated),
              'image/object/view':
              dataset_util.bytes_list_feature(poses),
              'image/caption/string':
              dataset_util.bytes_list_feature(caption_string),
              'image/caption/offset':
              dataset_util.int64_list_feature(caption_offset),
              'image/caption/length':
              dataset_util.int64_list_feature(caption_length),
              'image/proposal/bbox/ymin':
              dataset_util.float_list_feature(proposals[:, 0].tolist()),
              'image/proposal/bbox/xmin':
              dataset_util.float_list_feature(proposals[:, 1].tolist()),
              'image/proposal/bbox/ymax':
              dataset_util.float_list_feature(proposals[:, 2].tolist()),
              'image/proposal/bbox/xmax':
              dataset_util.float_list_feature(proposals[:, 3].tolist()),
          }))
  return example


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

  writers = []
  for i in range(FLAGS.number_of_parts):
    filename = FLAGS.output_path + '-%05d-of-%05d' % (i, FLAGS.number_of_parts)
    writers.append(tf.python_io.TFRecordWriter(filename))

  count = 0
  with tarfile.open(FLAGS.image_tar_file, "r:tar") as tar:
    for tarinfo in tar:
      if not tarinfo.isreg(): continue
      image_id = tarinfo.name.split('/')[1].split('.')[0]
      if not image_id.isdigit(): continue
      image_id = int(image_id)

      count += 1
      if count % 100 == 0:
        tf.logging.info('On %i: %i', count, image_id)

      fid = tar.extractfile(tarinfo)
      encoded_jpg = fid.read()

      tf_example = _create_tf_example(image_id, annotations[image_id],
                                      encoded_jpg)
      writers[count % FLAGS.number_of_parts].write(tf_example.SerializeToString())

  for writer in writers:
    writer.close()


if __name__ == '__main__':
  tf.app.run()
