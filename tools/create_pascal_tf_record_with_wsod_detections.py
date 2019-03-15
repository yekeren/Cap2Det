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
import json

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
flags.DEFINE_integer('number_of_parts', 20, 'Number of output parts.')
flags.DEFINE_string('wsod_detection_dir', None,
                    "Directory to wsod detection results.")

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']

cv2.setUseOptimized(True)
cv2.setNumThreads(4)


def filter_wsod_detections(data, classes, top_k=1):
  """Filters wsod detections by image-level object categories.

  Args:
    data: dict holding detection results from WSOD method.
    classes: set of ground truth image-level object classes.
    top_k: preserves top k detection box in each class.

  Returns:
    List of detected objects.
  """
  quota = {c: top_k for c in classes}
  results = []
  for obj in data:
    class_text = obj["category_id"]
    if class_text in classes and quota[class_text] > 0:
      x_min, y_min, width, height = obj["bbox"]
      x_max = x_min + width
      y_max = y_min + height

      obj["bndbox"] = {
          "xmin": x_min,
          "ymin": y_min,
          "xmax": x_max,
          "ymax": y_max,
      }
      obj["name"] = class_text
      results.append(obj)
      quota[class_text] -= 1
    if sum(quota.values()) == 0:
      break
  return results


def dict_to_tf_example(data,
                       wsod_detection_path,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalize_oicrs the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict).
    wsod_detection_path: dict holding detection results from WSOD method.
    dataset_directory: Path to root directory holding PASCAL dataset.
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

  # Load WSOD detection results.
  if not os.path.isfile(wsod_detection_path):
    raise ValueError("wsod detection result at %s is missing",
                     wsod_detection_path)

  with open(wsod_detection_path, "r") as fp:
    wsod_detections = json.load(fp)

  allowed_classes = set([])
  if 'object' in data:

    tf.logging.info("Original detections:\n%s",
                    json.dumps(data['object'], indent=2))

    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue
      allowed_classes.add(obj['name'])

  filtered_wsod_detections = filter_wsod_detections(
      wsod_detections, allowed_classes, top_k=1)

  tf.logging.info("Filtered detections:\n%s",
                  json.dumps(filtered_wsod_detections, indent=2))
  tf.logging.info("File: %s, Allowed classes: %s", data['filename'],
                  ", ".join(allowed_classes))
  tf.logging.info("=" * 40)

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  for obj in filtered_wsod_detections:
    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    classes.append(label_map_dict[obj['name']])
    classes_text.append(obj['name'].encode('utf8'))

    difficult_obj.append(0)
    truncated.append(0)
    poses.append("Frontal".encode("utf8"))

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
          }))
  return example


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
  if FLAGS.year not in YEARS:
    raise ValueError('year must be in : {}'.format(YEARS))
  if not os.path.isdir(FLAGS.wsod_detection_dir):
    raise ValueError('`wsod_detection_dir` is invalid.')

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
      wsod_detection_path = os.path.join(FLAGS.wsod_detection_dir,
                                         example + ".json")

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

      tf_example = dict_to_tf_example(data, wsod_detection_path, FLAGS.data_dir,
                                      label_map_dict,
                                      FLAGS.ignore_difficult_instances)
      writers[idx % FLAGS.number_of_parts].write(tf_example.SerializeToString())

  for writer in writers:
    writer.close()


if __name__ == '__main__':
  tf.app.run()
