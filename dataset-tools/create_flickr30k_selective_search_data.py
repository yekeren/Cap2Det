r"""Extract Selective Search proposals for the Flickr30k dataset.

Example usage:
    python create_flickr30k_selective_search_data.py \
        --image_tar_file=/home/user/VOCdevkit \
        --output_dir=/home/user/pascal.record
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
flags.DEFINE_string('image_tar_file', '', 'Path to the .tar file.')
flags.DEFINE_string('output_dir', '',
                    'Path to the directory saving output .npy files.')
tf.flags.DEFINE_string(
    'process_indicator', '0/1',
    'Process indicator, e.g. 2/5 denotes the 3rd process of the five')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

cv2.setUseOptimized(True)
cv2.setNumThreads(8)
ss_handler = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def main(_):
  process_id, num_processes = 0, 1
  if FLAGS.process_indicator:
    process_id, num_processes = [
        int(x) for x in FLAGS.process_indicator.split('/')
    ]
  tf.gfile.MakeDirs(FLAGS.output_dir)

  count = 0
  with tarfile.open(FLAGS.image_tar_file, "r:tar") as tar:
    for tarinfo in tar:
      if not tarinfo.isreg(): continue

      image_id = tarinfo.name.split('/')[1].split('.')[0]
      if not image_id.isdigit(): continue

      image_id = int(image_id)
      if image_id % num_processes != process_id:
        continue

      output_name = os.path.join(FLAGS.output_dir, '{}.npy'.format(image_id))
      if os.path.isfile(output_name):
        continue

      count += 1
      if count % 10 == 0:
        tf.logging.info('On %i: %i', count, image_id)
      fid = tar.extractfile(tarinfo)
      file_bytes = np.fromstring(fid.read(), dtype=np.uint8)
      bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

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

      x, y, w, h = [rects[:, i] for i in range(4)]
      proposals = np.stack(
          [y / height, x / width, (y + h) / height, (x + w) / width], axis=-1)

      with open(output_name, 'wb') as out_fid:
        np.save(out_fid, proposals)
  tf.logging.info('Done')


if __name__ == '__main__':
  tf.app.run()
