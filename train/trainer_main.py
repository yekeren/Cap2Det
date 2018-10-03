
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from google.protobuf import text_format

import reader
from protos import pipeline_pb2
from train import trainer

flags = tf.app.flags

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string('pipeline_proto', 
    'configs/gap448_lr0.005_reg_v5.pbtxt', 
    'Path to the pipeline proto file.')

FLAGS = flags.FLAGS


def load_pipeline_proto(filename):
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
  pipeline_proto = load_pipeline_proto(FLAGS.pipeline_proto)
  tf.logging.info("Pipeline configure: %s", '=' * 128)
  tf.logging.info(pipeline_proto)

  trainer.create_train_and_evaluate(pipeline_proto)

  tf.logging.info('Done')

if __name__ == '__main__':
  tf.app.run()
