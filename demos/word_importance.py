
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from protos import pipeline_pb2
from models import builder
import cv2
from core import plotlib
from core.standard_fields import InputDataFields
from core.standard_fields import GAPPredictions
from core.standard_fields import GAPPredictionTasks

flags = tf.app.flags

flags.DEFINE_string('pipeline_proto', 
    '', 'Path to the pipeline proto file.')

flags.DEFINE_bool('ascending_order', 
    False, 'If true, sort the word in ascending order.')

flags.DEFINE_integer('top_k',
    100, 'Number of top words to show.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


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
  tf.logging.info("Pipeline configure: %s", '=' * 128)
  tf.logging.info(pipeline_proto)

  g = tf.Graph()
  with g.as_default():

    # Infer saliency.

    model = builder.build(pipeline_proto.model, is_training=False)
    predictions = model.build_prediction(examples={}, 
        prediction_task=GAPPredictionTasks.word_saliency)

    vocabulary = predictions[GAPPredictions.vocabulary]
    saliency = predictions[GAPPredictions.word_saliency]

    saver = tf.train.Saver()
    invalid_variable_names = tf.report_uninitialized_variables()

  with tf.Session(graph=g) as sess:

    sess.run(tf.tables_initializer())

    # Load the latest checkpoint.

    checkpoint_path = tf.train.latest_checkpoint(pipeline_proto.model_dir)
    assert checkpoint_path is not None

    saver.restore(sess, checkpoint_path)
    assert len(sess.run(invalid_variable_names)) == 0

    # Print word importance.

    vocabulary, saliency = sess.run([vocabulary, saliency])
    if FLAGS.ascending_order:
      indices = np.argsort(saliency)
    else:
      indices = np.argsort(saliency)[::-1]

    for i in indices[:FLAGS.top_k]:
      tf.logging.info("%12s: %.4lf", vocabulary[i].decode('UTF-8'), saliency[i])
      print('%s\t%.4lf' % (vocabulary[i].decode('UTF-8'), saliency[i]))

  tf.logging.info('Done')

if __name__ == '__main__':
  tf.app.run()
