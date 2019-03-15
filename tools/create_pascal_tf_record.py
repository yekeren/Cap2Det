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
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
flags.DEFINE_string('selective_search_data', 'raw_data/selective_search_data',
                    'Path to label map proto')
flags.DEFINE_integer('number_of_parts', 20, 'Number of output parts.')
flags.DEFINE_boolean('normalize_oicr', False, 'Whether to normalize_oicr boxes')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']

cv2.setUseOptimized(True)
cv2.setNumThreads(4)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalize_oicrs the bounding box coordinates provided
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
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  # width = int(data['size']['width'])
  # height = int(data['size']['height'])
  height, width = image.height, image.width

  # Read SelectiveSearch to get the proposals.

  image_id = data['filename'].split('.')[0]
  filename = os.path.join(FLAGS.selective_search_data,
                          '{}.npy'.format(image_id))
  with open(filename, 'rb') as fid:
    #proposals = np.load(fid)[:2000, :]
    proposals = np.load(fid)

  if FLAGS.normalize_oicr:
    ymin, xmin, ymax, xmax = [proposals[:, i] for i in range(4)]
    ymin = ymin / height
    xmin = xmin / width
    ymax = ymax / height
    xmax = xmax / width
    proposals = np.stack([ymin, xmin, ymax, xmax], axis=-1)

  if proposals.shape[0] < 2000:
    tf.logging.info(proposals.shape)

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/height':
              dataset_util.int64_feature(height),
              'image/width':
              dataset_util.int64_feature(width),
              'image/filename':
              dataset_util.bytes_feature(data['filename'].encode('utf8')),
              'image/source_id':
              dataset_util.bytes_feature(data['filename'].encode('utf8')),
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
              dataset_util.bytes_list_feature(classes_text),
              'image/caption/offset':
              dataset_util.int64_list_feature([0]),
              'image/caption/length':
              dataset_util.int64_list_feature([len(classes_text)]),
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
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
  if FLAGS.year not in YEARS:
    raise ValueError('year must be in : {}'.format(YEARS))

  data_dir = FLAGS.data_dir
  years = ['VOC2007', 'VOC2012']
  if FLAGS.year != 'merged':
    years = [FLAGS.year]

  writers = []
  for i in range(FLAGS.number_of_parts):
    filename = FLAGS.output_path + '-%05d-of-%05d' % (i, FLAGS.number_of_parts)
    writers.append(tf.python_io.TFRecordWriter(filename))

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  for year in years:
    tf.logging.info('Reading from PASCAL %s dataset.', year)
    examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                                 'aeroplane_' + FLAGS.set + '.txt')
    annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)
    examples_list = dataset_util.read_examples_list(examples_path)
    for idx, example in enumerate(examples_list):
      if idx % 100 == 0:
        tf.logging.info('On image %d of %d', idx, len(examples_list))
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

      tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                      FLAGS.ignore_difficult_instances)
      writers[idx % FLAGS.number_of_parts].write(tf_example.SerializeToString())

  for writer in writers:
    writer.close()


if __name__ == '__main__':
  tf.app.run()
