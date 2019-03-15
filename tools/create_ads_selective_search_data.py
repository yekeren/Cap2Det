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

import os
import cv2
import tarfile
import numpy as np

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('image_data_dir', '', '')
flags.DEFINE_string('output_path', '', '')
tf.flags.DEFINE_string('parts', '', '')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

cv2.setUseOptimized(True)
cv2.setNumThreads(8)
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

  count = 0
  for (path, dirs, files) in os.walk(FLAGS.image_data_dir):
    for filename in files:
      if filename[-4:] in ['.jpg', '.png']:
        image_id = int(filename.split('.')[0])
        filename = os.path.join(path, filename)

      output_name = os.path.join(FLAGS.output_path, '{}.npy'.format(image_id))
      if os.path.isfile(output_name):
        tf.logging.info('%s is there.', output_name)
        continue

      if image_id % part_total == part_at_i:
        count += 1
        if count % 1 == 0:
          tf.logging.info('On %i: %i', count, image_id)

        bgr = cv2.imread(filename, cv2.IMREAD_COLOR)

        # Resize image if neccessary.
        height, width = bgr.shape[0], bgr.shape[1]
        if height / width >= 2.2:
          width = int(height / 2.2)
          bgr = cv2.resize(bgr, (width, height))
        elif width / height >= 2.2:
          height = int(width / 2.2)
          bgr = cv2.resize(bgr, (width, height))
        max_size = 500
        if max(height, width) > max_size:
          ratio = max(height, width) / max_size
          height = int(height / ratio)
          width = int(width/ ratio)
          bgr = cv2.resize(bgr, (width, height))
        height, width = bgr.shape[0], bgr.shape[1]

        ss.setBaseImage(bgr)
        #ss.switchToSelectiveSearchFast()
        ss.switchToSelectiveSearchQuality()
        rects = ss.process()

        rects = np.stack(
            [rect for rect in rects if rect[2] >= 20 and rect[3] >= 20], axis=0)

        x, y, w, h = [rects[:, i] for i in range(4)]
        proposals = np.stack(
            [y / height, x / width, (y + h) / height, (x + w) / width], axis=-1)

        # Write output file.
        with open(output_name, 'wb') as out_fid:
          np.save(out_fid, proposals)


if __name__ == '__main__':
  tf.app.run()
