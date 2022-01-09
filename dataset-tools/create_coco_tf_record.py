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

from pycocotools import mask
import tensorflow as tf
import collections

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
tf.flags.DEFINE_string('proposal_data_path', 'raw_data/coco_ssbox_quality',
                       'Directory to the proposal data.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def _process_caption(caption):
  """Processes a caption string into a list of tonenized words.

  Args:
    caption: A string caption.
  Returns:
    A list of strings; the tokenized caption.
  """
  return nltk.tokenize.word_tokenize(caption.lower())


def create_tf_example(image,
                      annotations_list,
                      caption_annotations_list,
                      zip_handler,
                      category_index,
                      include_masks=False,
                      subdir=None,
                      proposal_data_path=None):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official COCO dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    caption_annotations_list:
      list of dicts with keys:
      [u'image_id', u'id', u'caption']
    zip_handler: class for reading and writing ZIP files.
    category_index: a dict containing COCO category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  with zip_handler.open(subdir + '/' + filename, "r") as fid:
    encoded_jpg = fid.read()

  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  key = hashlib.sha256(encoded_jpg).hexdigest()

  npy_path = os.path.join(proposal_data_path,
                          '{}/{}.npy'.format(image_id % 10, image_id))

  with open(npy_path, 'rb') as fid:
    proposals = np.load(fid)

  xmin = []
  xmax = []
  ymin = []
  ymax = []
  is_crowd = []
  category_names = []
  category_ids = []
  area = []
  encoded_mask_png = []
  num_annotations_skipped = 0
  for object_annotations in annotations_list:
    (x, y, width, height) = tuple(object_annotations['bbox'])
    if width <= 0 or height <= 0:
      num_annotations_skipped += 1
      continue
    if x + width > image_width or y + height > image_height:
      num_annotations_skipped += 1
      continue
    xmin.append(float(x) / image_width)
    xmax.append(float(x + width) / image_width)
    ymin.append(float(y) / image_height)
    ymax.append(float(y + height) / image_height)
    is_crowd.append(object_annotations['iscrowd'])
    category_id = int(object_annotations['category_id'])
    category_ids.append(category_id)
    category_names.append(category_index[category_id]['name'].encode('utf8'))
    area.append(object_annotations['area'])

    if include_masks:
      run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                          image_height, image_width)
      binary_mask = mask.decode(run_len_encoding)
      if not object_annotations['iscrowd']:
        binary_mask = np.amax(binary_mask, axis=2)
      pil_image = PIL.Image.fromarray(binary_mask)
      output_io = io.BytesIO()
      pil_image.save(output_io, format='PNG')
      encoded_mask_png.append(output_io.getvalue())

  caption_string = []
  caption_offset = []
  caption_length = []
  for caption_annotations in caption_annotations_list:
    caption = _process_caption(caption_annotations['caption'])
    caption_offset.append(len(caption_string))
    caption_length.append(len(caption))
    caption_string.extend(caption)
  caption_string = [caption.encode('utf8') for caption in caption_string]

  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
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
      'image/object/class/label':
          dataset_util.int64_list_feature(category_ids),
      'image/object/class/text':
          dataset_util.bytes_list_feature(category_names),
      'image/object/is_crowd':
          dataset_util.int64_list_feature(is_crowd),
      'image/object/area':
          dataset_util.float_list_feature(area),
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
  }
  if include_masks:
    feature_dict['image/object/mask'] = (
        dataset_util.bytes_list_feature(encoded_mask_png))
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return key, example, num_annotations_skipped


def _create_tf_record_from_coco_annotations(annotations_file,
                                            caption_annotations_file, zip_file,
                                            output_path, include_masks,
                                            num_shards, proposal_data_path):
  """Loads COCO annotation json files and converts to tf.Record format.

  Args:
    annotations_file: JSON file containing bounding box annotations.
    caption_annotations_file: JSON file containing caption annotations.
    zip_file: ZIP file containing the image files.
    output_path: Path to output tf.Record file.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    num_shards: number of output file shards.
  """
  with contextlib2.ExitStack() as tf_record_close_stack, \
      tf.gfile.GFile(annotations_file, 'r') as fid:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_path, num_shards)
    groundtruth_data = json.load(fid)
    images = groundtruth_data['images']
    category_index = label_map_util.create_category_index(
        groundtruth_data['categories'])

    annotations_index = {}
    if 'annotations' in groundtruth_data:
      tf.logging.info(
          'Found groundtruth annotations. Building annotations index.')
      for annotation in groundtruth_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_index:
          annotations_index[image_id] = []
        annotations_index[image_id].append(annotation)

    caption_annotations_index = {}
    if caption_annotations_file:
      with tf.gfile.GFile(caption_annotations_file, 'r') as cap_fid:
        caption_groundtruth_data = json.load(cap_fid)

        assert (groundtruth_data['images'] == caption_groundtruth_data['images']
               ), 'The detection and caption sets are different.'

        if 'annotations' in caption_groundtruth_data:
          tf.logging.info(
              'Found caption groundtruth annotations. Building annotations index.'
          )
          for annotation in caption_groundtruth_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in caption_annotations_index:
              caption_annotations_index[image_id] = []
            caption_annotations_index[image_id].append(annotation)

    missing_annotation_count = 0
    missing_caption_annotation_count = 0
    for image in images:
      image_id = image['id']
      if image_id not in annotations_index:
        missing_annotation_count += 1
        annotations_index[image_id] = []
      if image_id not in caption_annotations_index:
        missing_caption_annotation_count += 1
        caption_annotations_index[image_id] = []
    tf.logging.info('%d images are missing annotations.',
                    missing_annotation_count)
    tf.logging.info('%d images are missing caption annotations.',
                    missing_caption_annotation_count)

    with zipfile.ZipFile(zip_file) as zip_handler:
      total_num_annotations_skipped = 0
      for idx, image in enumerate(images):
        if idx % 100 == 0:
          tf.logging.info('On image %d of %d', idx, len(images))
        annotations_list = annotations_index[image['id']]
        caption_annotations_list = caption_annotations_index[image['id']]
        subdir = image['coco_url'].split('/')[-2]
        _, tf_example, num_annotations_skipped = create_tf_example(
            image, annotations_list, caption_annotations_list, zip_handler,
            category_index, include_masks, subdir, proposal_data_path)
        if tf_example is not None:
          total_num_annotations_skipped += num_annotations_skipped
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
        else:
          tf.logging.info('Filtered by pascal classes.')
      tf.logging.info('Finished writing, skipped %d annotations.',
                      total_num_annotations_skipped)


def main(_):
  assert FLAGS.train_image_file, '`train_image_file` missing.'
  assert FLAGS.val_image_file, '`val_image_file` missing.'
  assert FLAGS.test_image_file, '`test_image_file` missing.'
  assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
  assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
  assert FLAGS.testdev_annotations_file, '`testdev_annotations_file` missing.'
  assert FLAGS.train_caption_annotations_file, '`train_caption_annotations_file` missing.'
  assert FLAGS.val_caption_annotations_file, '`val_caption_annotations_file` missing.'

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  train_output_path = os.path.join(FLAGS.output_dir, 'coco17_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'coco17_val.record')
  testdev_output_path = os.path.join(FLAGS.output_dir, 'coco17_testdev.record')

  _create_tf_record_from_coco_annotations(
      FLAGS.train_annotations_file,
      FLAGS.train_caption_annotations_file,
      FLAGS.train_image_file,
      train_output_path,
      FLAGS.include_masks,
      num_shards=100,
      proposal_data_path=FLAGS.proposal_data_path)
  _create_tf_record_from_coco_annotations(
      FLAGS.val_annotations_file,
      FLAGS.val_caption_annotations_file,
      FLAGS.val_image_file,
      val_output_path,
      FLAGS.include_masks,
      num_shards=5,
      proposal_data_path=FLAGS.proposal_data_path)
  _create_tf_record_from_coco_annotations(
      FLAGS.testdev_annotations_file,
      None,
      FLAGS.test_image_file,
      testdev_output_path,
      FLAGS.include_masks,
      num_shards=50,
      proposal_data_path=FLAGS.proposal_data_path)


if __name__ == '__main__':
  tf.app.run()
