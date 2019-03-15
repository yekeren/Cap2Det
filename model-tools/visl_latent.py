from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json
import cv2
import numpy as np
import collections
import tensorflow as tf
from google.protobuf import text_format

from core.standard_fields import InputDataFields
from core.standard_fields import DetectionResultFields
from train import trainer
from protos import pipeline_pb2
from core.plotlib import _py_convert_to_heatmap

from object_detection.utils import object_detection_evaluation

flags = tf.app.flags

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string('pipeline_proto', '', 'Path to the pipeline proto file.')

flags.DEFINE_string('model_dir', '',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('vocabulary_file', '',
                    'Path to the detection vocabulary file.')

flags.DEFINE_string('result_dir', 'results',
                    'Path to the directory saving results.')

FLAGS = flags.FLAGS

_EPSILON = 1e-10
_PIXELS_PER_GRID = 48


def _load_pipeline_proto(filename):
  """Loads pipeline proto from file.

  Args:
    filename: path to the pipeline config file.

  Returns:
    an instance of pipeline_pb2.Pipeline.
  """
  pipeline_proto = pipeline_pb2.Pipeline()
  with tf.gfile.GFile(filename, 'r') as fp:
    text_format.Merge(fp.read(), pipeline_proto)
  return pipeline_proto


def _analyze_data(proba_h_given_c, category_to_id, categories):
  """Runs the prediction.

  Args:
    proba_h_given_c: A [num_latent_factors, num_classes] numpy array.
    category_to_id: A python dict maps from the category name to integer id.
    categories: A python list storing class names.
  """

  data = proba_h_given_c.transpose()
  num_classes, num_latent_factors = data.shape

  # Resize the heatmap.

  heatmap = _py_convert_to_heatmap(1.0 - data, normalize=True, cmap="jet")
  new_height = _PIXELS_PER_GRID * num_classes
  new_width = _PIXELS_PER_GRID * num_latent_factors

  heatmap = cv2.resize(
      heatmap, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
  heatmap = (heatmap * 255.0).astype(np.uint8)

  # Add categorical axis.

  x_axis = np.zeros(
      ((_PIXELS_PER_GRID, _PIXELS_PER_GRID * num_latent_factors, 3)),
      dtype=np.uint8)
  y_axis = np.zeros(
      ((_PIXELS_PER_GRID * (1 + num_classes), _PIXELS_PER_GRID, 3)),
      dtype=np.uint8)

  for i, catInfo in enumerate(categories):
    #cv2.putText(
    #    x_axis,
    #    catInfo['name'],
    #    org=(_PIXELS_PER_GRID * i, 16),
    #    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
    #    fontScale=0.5,
    #    color=(255, 255, 255),
    #    thickness=1)
    cv2.putText(
        y_axis,
        catInfo['name'],
        org=(0, _PIXELS_PER_GRID * (1 + i) + 16),
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        fontScale=0.5,
        color=(255, 255, 255),
        thickness=1)

  heatmap = np.concatenate([x_axis, heatmap], axis=0)
  heatmap = np.concatenate([y_axis, heatmap], axis=1)

  return heatmap


def main(_):
  pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)

  if FLAGS.model_dir:
    pipeline_proto.model_dir = FLAGS.model_dir
    tf.logging.info("Override model checkpoint dir: %s", FLAGS.model_dir)
  tf.logging.info("Pipeline configure: %s", '=' * 128)
  tf.logging.info(pipeline_proto)

  # Load the vocabulary file.

  categories = []
  category_to_id = {}
  with open(FLAGS.vocabulary_file, 'r') as fp:
    for line_id, line in enumerate(fp.readlines()):
      categories.append({'id': 1 + line_id, 'name': line.strip('\n')})
      category_to_id[line.strip('\n')] = 1 + line_id
  tf.logging.info("\n%s", json.dumps(categories, indent=2))

  # Start to predict.

  checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
  assert checkpoint_path is not None

  global_step = int(checkpoint_path.split('-')[-1])
  for examples in trainer.predict(pipeline_proto, checkpoint_path):
    if 'midn_proba_h_given_c' in examples:
      heatmap = _analyze_data(examples['midn_proba_h_given_c'], category_to_id,
                              categories)
      filename = FLAGS.pipeline_proto.split('/')[1].split(
          '.')[0] + '_{}.jpg'.format(global_step)
      filename = os.path.join(FLAGS.result_dir, filename)
      cv2.imwrite(filename, heatmap)
      tf.logging.info('Results are written to %s', filename)
    break

  tf.logging.info('Done')


if __name__ == '__main__':
  tf.app.run()
