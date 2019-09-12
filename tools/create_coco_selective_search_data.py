r"""Extract Selective Search proposals for the MSCOCO dataset.

Example usage:
    python create_coco_selective_search_data.py \
      --logtostderr \
      --train_image_file="${TRAIN_IMAGE_FILE}" \
      --val_image_file="${VAL_IMAGE_FILE}" \
      --test_image_file="${TEST_IMAGE_FILE}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import json
import zipfile

import numpy as np

import tensorflow as tf

flags = tf.app.flags
tf.flags.DEFINE_string('train_image_file', '', 'Training image zip file.')
tf.flags.DEFINE_string('val_image_file', '', 'Validation image zip file.')
tf.flags.DEFINE_string('test_image_file', '', 'Test image zip file.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('testdev_annotations_file', '',
                       'Test-dev annotations JSON file.')
tf.flags.DEFINE_string('output_dir', 'raw_data/proposal_data',
                       'Output data directory.')
tf.flags.DEFINE_string(
    'process_indicator', '0/1',
    'Process indicator, e.g. 2/5 denotes the 3rd process of the five')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

cv2.setUseOptimized(True)
cv2.setNumThreads(4)
ss_handler = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def _create_ssbox_data(annotations_file, zip_file, output_path, process_id,
                       num_processes):
  """Loads COCO annotation json files and converts to tf.Record format.

  Args:
    annotations_file: JSON file containing bounding box annotations.
    zip_file: ZIP file containing the image files.
    output_path: Path to output tf.Record file.
  """
  # Read the annotation file.

  with tf.gfile.GFile(annotations_file, 'r') as fid:
    groundtruth_data = json.load(fid)
  images = groundtruth_data['images']

  # Extract the region proposals.

  with zipfile.ZipFile(zip_file) as zip_handler:

    for index, image in enumerate(images):
      if index % 100 == 0:
        tf.logging.info('On image %d of %d', index, len(images))

      image_id = image['id']
      filename = image['file_name']
      coco_url = image['coco_url']
      if image_id % num_processes != process_id:
        continue

      output_name = os.path.join(output_path, '{}/{}.npy'.format(
          image_id % 10, image_id))
      if os.path.isfile(output_name):
        continue

      # OpenCV Selective Search algorithm.

      sub_dir = coco_url.split('/')[-2]
      with zip_handler.open(sub_dir + '/' + filename, "r") as fid:
        encoded_jpg = fid.read()
      file_bytes = np.fromstring(encoded_jpg, dtype=np.uint8)
      bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

      # Resize image.

      height, width = bgr.shape[0], bgr.shape[1]
      if height / width >= 2.2:
        width = int(height / 2.2)
        bgr = cv2.resize(bgr, (width, height))
      elif width / height >= 2.2:
        height = int(width / 2.2)
        bgr = cv2.resize(bgr, (width, height))
      height, width = bgr.shape[0], bgr.shape[1]

      ss_handler.setBaseImage(bgr)
      ss_handler.switchToSelectiveSearchQuality()
      rects = ss_handler.process()
      rects = np.stack(
          [rect for rect in rects if rect[2] >= 20 and rect[3] >= 20], axis=0)

      # Normalize the proposals.

      x, y, w, h = [rects[:, i] for i in range(4)]
      proposals = np.stack(
          [y / height, x / width, (y + h) / height, (x + w) / width], axis=-1)

      with open(output_name, 'wb') as out_fid:
        np.save(out_fid, proposals)


def main(_):
  process_id, num_processes = 0, 1
  if FLAGS.process_indicator:
    process_id, num_processes = [
        int(x) for x in FLAGS.process_indicator.split('/')
    ]

  for i in range(10):
    tf.gfile.MakeDirs(os.path.join(FLAGS.output_dir, '%i' % i))

  _create_ssbox_data(FLAGS.train_annotations_file, FLAGS.train_image_file,
                     FLAGS.output_dir, process_id, num_processes)
  _create_ssbox_data(FLAGS.val_annotations_file, FLAGS.val_image_file,
                     FLAGS.output_dir, process_id, num_processes)
  _create_ssbox_data(FLAGS.testdev_annotations_file, FLAGS.test_image_file,
                     FLAGS.output_dir, process_id, num_processes)


if __name__ == '__main__':
  tf.app.run()
