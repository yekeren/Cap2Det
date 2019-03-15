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
r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os

import cv2
import numpy as np
from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from core import imgproc

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
flags.DEFINE_string('output_data_path', 'raw_data/selective_search_data',
                    'Path to label map proto')
tf.flags.DEFINE_string('parts', '', '')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']

cv2.setUseOptimized(True)
cv2.setNumThreads(4)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def selective_search(data,
                     dataset_directory,
                     label_map_dict,
                     ignore_difficult_instances=False,
                     image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  # Use SelectiveSearch to get the proposals.

  file_bytes = np.fromstring(encoded_jpg, dtype=np.uint8)

  bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
  height, width, _ = bgr.shape
  assert bgr.shape[0] == height and bgr.shape[1] == width

  if height / width >= 2.2:
    width = int(height / 2.2)
    bgr = cv2.resize(bgr, (width, height))
  elif width / height >= 2.2:
    height = int(width / 2.2)
    bgr = cv2.resize(bgr, (width, height))

  ss.setBaseImage(bgr)
  #ss.switchToSelectiveSearchFast()
  ss.switchToSelectiveSearchQuality()
  rects = ss.process()

  rects = np.stack([rect for rect in rects if rect[2] >= 20 and rect[3] >= 20],
                   axis=0)

  height, width = bgr.shape[0], bgr.shape[1]
  x, y, w, h = [rects[:, i] for i in range(4)]
  proposals = np.stack(
      [y / height, x / width, (y + h) / height, (x + w) / width], axis=-1)
  return proposals


def main(_):
  part_at_i, part_total = 0, 1
  if FLAGS.parts:
    part_at_i, part_total = [int(x) for x in FLAGS.parts.split('/')]

  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
  if FLAGS.year not in YEARS:
    raise ValueError('year must be in : {}'.format(YEARS))

  data_dir = FLAGS.data_dir
  years = ['VOC2007', 'VOC2012']
  if FLAGS.year != 'merged':
    years = [FLAGS.year]

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  for year in years:
    tf.logging.info('Reading from PASCAL %s dataset.', year)
    examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                                 'aeroplane_' + FLAGS.set + '.txt')
    annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)
    examples_list = dataset_util.read_examples_list(examples_path)
    count = 0
    for idx, example in enumerate(examples_list):
      count += 1

      path = os.path.join(annotations_dir, example + '.xml')

      if os.path.isfile(path):

        # For trainval example.

        with tf.gfile.GFile(path, 'r') as fid:
          xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

      else:

        # For test example.

        filename = example + '.jpg'

        data = {
            'folder': 'VOC2012',
            'filename': filename,
            'source': {
                'database': 'The VOC2008 Database',
                'annotation': 'PASCAL VOC2008',
                'image': 'flickr'
            },
            'size': {
                'width': '99999',
                'height': '99999',
                'depth': '3'
            },
            'segmented': '0',
            'object': []
        }

      image_id = example
      if int(image_id) % part_total == part_at_i:
        filename = os.path.join(FLAGS.output_data_path,
                                '{}.npy'.format(image_id))
        if not os.path.isfile(filename):
          rects = selective_search(data, FLAGS.data_dir, label_map_dict,
                                   FLAGS.ignore_difficult_instances)
          with open(filename, 'wb') as fid:
            np.save(fid, rects)
          tf.logging.info('On image %d of %d, %s', idx, len(examples_list),
                          image_id)
        else:
          tf.logging.info('%s is there', filename)

    tf.logging.info("Total: %i", count)


if __name__ == '__main__':
  tf.app.run()
