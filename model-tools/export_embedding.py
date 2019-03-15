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

#flags.DEFINE_string('pipeline_proto', 'configs.iccv.coco/visual_w2v_v12.pbtxt',
#                    'Path to the pipeline proto file.')
#
#flags.DEFINE_string('model_dir', 'ICCV-TXT-logs/visual_w2v_v12',
#                    'Path to the directory which holds model checkpoints.')
#
#flags.DEFINE_string('output_path', 'configs/coco_open_vocab_50d_learned.npy',
#                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('pipeline_proto', 'configs.iccv.flickr30k/visual_w2v_flickr30k.pbtxt',
                    'Path to the pipeline proto file.')

flags.DEFINE_string('model_dir', 'ICCV-TXT-logs/visual_w2v_flickr30k',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('output_path', 'configs/flickr30k_open_vocab_50d_learned.npy',
                    'Path to the directory which holds model checkpoints.')
FLAGS = flags.FLAGS


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


def main(_):
  pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)

  if FLAGS.model_dir:
    pipeline_proto.model_dir = FLAGS.model_dir
    tf.logging.info("Override model checkpoint dir: %s", FLAGS.model_dir)
  tf.logging.info("Pipeline configure: %s", '=' * 128)
  tf.logging.info(pipeline_proto)

  # Start to predict.

  checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
  assert checkpoint_path is not None

  global_step = int(checkpoint_path.split('-')[-1])
  for examples in trainer.predict(pipeline_proto, checkpoint_path):
    if 'word2vec' in examples:
      np.save(FLAGS.output_path, examples['word2vec'])
      tf.logging.info('Results are written to %s', FLAGS.output_path)
    break

  tf.logging.info('Done')


if __name__ == '__main__':
  tf.app.run()
