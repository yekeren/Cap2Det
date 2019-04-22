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
import cv2

import tensorflow as tf
import collections

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags

flags.DEFINE_string('image_data_dir', '', '')
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


no_because = 0
n_stmts = 0


def _create_tf_example(image_id, qa_data, encoded_jpg):
  # encoded_jpg_io = io.BytesIO(encoded_jpg)
  # image = PIL.Image.open(encoded_jpg_io)
  # height, width = image.height, image.width
  # key = hashlib.sha256(encoded_jpg).hexdigest()

  file_bytes = np.fromstring(encoded_jpg, dtype=np.uint8)
  bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  height, width, _ = bgr.shape

  max_size = max(height, width)
  if max_size > 500:
    ratio = max_size / 500
    height = int(height / ratio)
    width = int(width / ratio)
    bgr = cv2.resize(bgr, (width, height))
    encoded_jpg = cv2.imencode('.jpg', bgr)[1].tostring()

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

  global no_because
  global n_stmts

  captions = qa_data['groundtruth_list']
  actions, reasons = [], []
  for caption in captions:
    pos = caption.lower().replace('beacause', 'because').replace(
        'becasue', 'because').replace('becuase', 'because').replace(
            'becaise', 'because').find('because')
    if pos >= 0:
      actions.append(caption[:pos])
      reasons.append(caption[pos + len('because'):])
    else:
      #actions.append(caption)
      #reasons.append(caption)
      tf.logging.info(caption)
      no_because += 1
    n_stmts += 1

  # Process action.
  caption_string = []
  caption_offset = []
  caption_length = []
  for caption in actions:
    caption = _process_caption(caption)
    caption_offset.append(len(caption_string))
    caption_length.append(len(caption))
    caption_string.extend(caption)
  caption_string = [caption.encode('utf8') for caption in caption_string]
  (action_string, action_offset,
   action_length) = (caption_string, caption_offset, caption_length)

  # Process reason.
  caption_string = []
  caption_offset = []
  caption_length = []
  for caption in reasons:
    caption = _process_caption(caption)
    caption_offset.append(len(caption_string))
    caption_length.append(len(caption))
    caption_string.extend(caption)
  caption_string = [caption.encode('utf8') for caption in caption_string]
  (reason_string, reason_offset,
   reason_length) = (caption_string, caption_offset, caption_length)

  # Process action-reason.
  caption_string = []
  caption_offset = []
  caption_length = []
  for caption in captions:
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
              'image/action/string':
              dataset_util.bytes_list_feature(action_string),
              'image/action/offset':
              dataset_util.int64_list_feature(action_offset),
              'image/action/length':
              dataset_util.int64_list_feature(action_length),
              'image/reason/string':
              dataset_util.bytes_list_feature(reason_string),
              'image/reason/offset':
              dataset_util.int64_list_feature(reason_offset),
              'image/reason/length':
              dataset_util.int64_list_feature(reason_length),
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


def main(_):
  writers = []
  for i in range(FLAGS.number_of_parts):
    filename = FLAGS.output_path + '-%05d-of-%05d' % (i, FLAGS.number_of_parts)
    writers.append(tf.python_io.TFRecordWriter(filename))

  count = 0
  for root, dirs, files in os.walk(FLAGS.image_data_dir):
    for filename in files:
      image_id = filename.split('.')[0]

      filename = os.path.join(root, filename)
      with open(filename, 'rb') as fid:
        encoded_jpg = fid.read()

      filename = os.path.join(FLAGS.annotation_path, '%s.json' % (image_id))
      with open(filename, 'r') as fid:
        qa_data = json.load(fid)

      count += 1
      if count % 100 == 0:
        tf.logging.info('On %i: %s', count, image_id)

      tf_example = _create_tf_example(image_id, qa_data, encoded_jpg)
      writers[count % FLAGS.number_of_parts].write(
          tf_example.SerializeToString())

  for writer in writers:
    writer.close()

  tf.logging.info('invalid ratio: %i/%i', no_because, n_stmts)


if __name__ == '__main__':
  tf.app.run()
