from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from protos import pipeline_pb2

from train import trainer

from tensorflow.python.platform import tf_logging as logging

from sklearn.metrics import average_precision_score

flags = tf.app.flags

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string('pipeline_proto', '', 'Path to the pipeline proto file.')

flags.DEFINE_string('model_dir', '',
                    'Path to the directory which holds model checkpoints.')

FLAGS = flags.FLAGS

try:
  logging._get_logger().propagate = False
except AttributeError:
  pass


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

  checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
  tf.logging.info('Start to evaluate checkpoint %s.', checkpoint_path)

  y_true, y_pred= [], []
  for batch_id, examples in enumerate(
      trainer.predict(pipeline_proto, checkpoint_path)):
    vocab, logits, object_names = (examples['vocab'], examples['logits'],
                                   examples['object_texts'])
    vocab = vocab.tolist()
    labels = np.zeros_like(logits)
    assert labels.shape[0] == 1

    for name in object_names[0]:
      labels[0, vocab.index(name)] = 1.0

    y_true.append(labels)
    y_pred.append(logits)

  y_true = np.concatenate(y_true, axis=0)
  y_pred = np.concatenate(y_pred, axis=0)

  mAP = average_precision_score(y_true, y_pred, average='micro')
  tf.logging.info('Evaluated %i examples.', batch_id + 1)
  tf.logging.info('Final mAP is %.3lf', mAP)


if __name__ == '__main__':
  tf.app.run()
