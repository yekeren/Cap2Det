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

from pycocotools import mask
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
tf.flags.DEFINE_boolean(
    'include_masks', False, 'Whether to include instance segmentations masks '
    '(PNG encoded) in the result. default: False.')
tf.flags.DEFINE_string('train_image_file', '', 'Training image zip file.')
tf.flags.DEFINE_string('val_image_file', '', 'Validation image zip file.')
tf.flags.DEFINE_string('test_image_file', '', 'Test image zip file.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('train_caption_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('val_caption_annotations_file', '',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('testdev_annotations_file', '',
                       'Test-dev annotations JSON file.')
tf.flags.DEFINE_string('output_dir', 'output/', 'Output data directory.')
tf.flags.DEFINE_string('structured_edge_detection_model',
                       'zoo/ximgproc/model.yml', '')
tf.flags.DEFINE_string('parts', '', '')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

cv2.setUseOptimized(True)
cv2.setNumThreads(4)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def _create_ssbox_data_from_coco_annotations(annotations_file, zip_file,
                                             output_path):
  """Loads COCO annotation json files and converts to tf.Record format.

  Args:
    annotations_file: JSON file containing bounding box annotations.
    zip_file: ZIP file containing the image files.
    output_path: Path to output tf.Record file.
  """
  part_at_i, part_total = 0, 1
  if FLAGS.parts:
    part_at_i, part_total = [int(x) for x in FLAGS.parts.split('/')]

  with tf.gfile.GFile(annotations_file, 'r') as fid, \
      zipfile.ZipFile(zip_file) as zip_handler:
    groundtruth_data = json.load(fid)
    images = groundtruth_data['images']

    for index, image in enumerate(images):
      if index % 100 == 0:
        tf.logging.info('On image %d of %d', index, len(images))

      image_id = image['id']
      filename = image['file_name']

      if image_id % part_total == part_at_i:
        sub_dir = filename.split('_')[1]
        with zip_handler.open(sub_dir + '/' + filename, "r") as fid:
          encoded_jpg = fid.read()

        # OpenCV EdgeBoxes.

        file_bytes = np.fromstring(encoded_jpg, dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Resize image if neccessary.
        height, width = bgr.shape[0], bgr.shape[1]
        if height / width >= 2.2:
          width = int(height / 2.2)
          bgr = cv2.resize(bgr, (width, height))
        elif width / height >= 2.2:
          height = int(width / 2.2)
          bgr = cv2.resize(bgr, (width, height))
        height, width = bgr.shape[0], bgr.shape[1]

        ss.setBaseImage(bgr)
        #ss.switchToSelectiveSearchFast()
        ss.switchToSelectiveSearchQuality()
        rects = ss.process()

        rects = np.stack([rect for rect in rects if rect[2] >= 20 and rect[3] >= 20],
                         axis=0)

        x, y, w, h = [rects[:, i] for i in range(4)]
        proposals = np.stack(
            [y / height, x / width, (y + h) / height, (x + w) / width], axis=-1)

        output_name = os.path.join(output_path, '{}/{}.npy'.format(
            image_id % 10, image_id))
        with open(output_name, 'wb') as out_fid:
          np.save(out_fid, proposals)


def main(_):
  assert FLAGS.train_image_file, '`train_image_file` missing.'
  assert FLAGS.val_image_file, '`val_image_file` missing.'
  assert FLAGS.test_image_file, '`test_image_file` missing.'
  assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
  assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
  assert FLAGS.testdev_annotations_file, '`testdev_annotations_file` missing.'

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  _create_ssbox_data_from_coco_annotations(
      FLAGS.train_annotations_file, FLAGS.train_image_file, FLAGS.output_dir)
  _create_ssbox_data_from_coco_annotations(
      FLAGS.val_annotations_file, FLAGS.val_image_file, FLAGS.output_dir)
  _create_ssbox_data_from_coco_annotations(
      FLAGS.testdev_annotations_file, FLAGS.test_image_file, FLAGS.output_dir)


if __name__ == '__main__':
  tf.app.run()
