from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import cv2
import numpy as np
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import logging
from slim.nets import nets_factory

flags.DEFINE_string('image_data_path', 'raw_data/train_images/',
                    'Path to the directory saving images.')

flags.DEFINE_string('bounding_box_json_path', 'raw_data/densecap.json/',
                    'Path to the json annotation file.')

flags.DEFINE_string('feature_extractor_name', 'inception_v4',
                    'The name of the feature extractor.')

flags.DEFINE_string('feature_extractor_scope', 'InceptionV4',
                    'The variable scope of the feature extractor.')

flags.DEFINE_string('feature_extractor_endpoint', 'PreLogitsFlatten',
                    'The endpoint of the feature extractor.')

flags.DEFINE_string('feature_extractor_checkpoint', 'zoo/inception_v4.ckpt',
                    'The path to the checkpoint file.')

flags.DEFINE_string('feature_output_path', 'raw_data/roi.npy/',
                    'Path to the directory saving output image feature files.')

flags.DEFINE_integer('max_number_of_regions', 10, 'Maximum number of regions.')

FLAGS = flags.FLAGS
slim = tf.contrib.slim


def _load_image_path_list(image_data_path):
  """Loads image paths from the image_data_path.

  Args:
    image_data_path: path to the directory saving images.

  Returns:
    examples: a list of (image_id, filename) tuples.
  """
  examples = []
  for dirpath, dirnames, filenames in os.walk(image_data_path):
    for filename in filenames:
      image_id = int(filename.split('.')[0])
      filename = os.path.join(dirpath, filename)
      examples.append((image_id, filename))
  return examples


def _load_bounding_box_annotations(image_id, bounding_box_json_path):
  """Loads bounding box info.

  Args:
    image_id: image id.
    bounding_box_json_path: Path to the json annotation file.
  """
  filename = os.path.join(bounding_box_json_path, '{}.json'.format(image_id))
  with open(filename, 'r') as fid:
    return json.load(fid)


def _save(image_id, features, feature_output_path):
  """Saves image features to file.

  Args:
    image_id: image id.
    features: ROIs' features.
    feature_output_path: Path to the directory saving image features.
  """
  filename = os.path.join(feature_output_path, '{}.npy'.format(image_id))
  with open(filename, 'wb') as fid:
    np.save(fid, features)


def _crop_and_resize(image, bounding_box, crop_size):
  """Crops roi from an image and resizes it.

  Args:
    image: the image data.
    bounding_box: [ymin, xmin, ymax, xmax] representing the normalized bounding box.
    crop_size: the expected output size of the roi image.

  Returns:
    a [crop_size, crop_size, 3] roi image.
  """
  height, width, _ = image.shape

  ymin, xmin, ymax, xmax = bounding_box
  ymin = int(ymin * height)
  xmin = int(xmin * width)
  ymax = int(ymax * height)
  xmax = int(xmax * width)

  return cv2.resize(image[ymin:ymax, xmin:xmax, :], crop_size)


def main(_):
  logging.set_verbosity(logging.INFO)

  examples = _load_image_path_list(FLAGS.image_data_path)
  logging.info('Load %s examples.', len(examples))

  # Create computational graph.
  g = tf.Graph()
  with g.as_default():
    net_fn = nets_factory.get_network_fn(
        name=FLAGS.feature_extractor_name, num_classes=1001)
    default_image_size = getattr(net_fn, 'default_image_size', 224)

    images = tf.placeholder(
        shape=(None, default_image_size, default_image_size, 3),
        dtype=tf.float32)

    _, end_points = net_fn(images)
    output_tensor = end_points[FLAGS.feature_extractor_endpoint]

    init_fn = slim.assign_from_checkpoint_fn(
        FLAGS.feature_extractor_checkpoint,
        slim.get_model_variables(FLAGS.feature_extractor_scope))
    uninitialized_variable_names = tf.report_uninitialized_variables()

  # Start session.
  with tf.Session(graph=g) as sess:
    init_fn(sess)
    assert len(sess.run(uninitialized_variable_names)) == 0

    for index, (image_id, filename) in enumerate(examples):
      if index % 10 == 0:
        logging.info('On image %i/%i', index, len(examples))

      # Load bounding boxes.
      data = _load_bounding_box_annotations(image_id,
                                            FLAGS.bounding_box_json_path)

      # Load image, preprocess.
      bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
      rgb = bgr[:, :, ::-1].astype(np.float32) * 2.0 / 255.0 - 1.0

      # Batch all the ROIs within an image.
      rois = []
      for paragraph in data['paragraphs'][:FLAGS.max_number_of_regions]:
        roi = _crop_and_resize(
            rgb,
            bounding_box=(paragraph['bounding_box']['ymin'],
                          paragraph['bounding_box']['xmin'],
                          paragraph['bounding_box']['ymax'],
                          paragraph['bounding_box']['xmax']),
            crop_size=(default_image_size, default_image_size))
        rois.append(roi)
      batch = np.stack(rois, axis=0)

      features = sess.run(output_tensor, feed_dict={images: batch})
      _save(image_id, features, FLAGS.feature_output_path)

  logging.info('Done')


if __name__ == '__main__':
  app.run()
