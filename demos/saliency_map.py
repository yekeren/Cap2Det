
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

flags.DEFINE_string('image_path', 
    'testdata', 
    'Path to the directory storing image files.')

flags.DEFINE_string('demo_path', 
    'tmp', 
    'Path to the directory storing demo results.')

FLAGS = flags.FLAGS

_SMALL_NUMBER = 1e-8

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


def _get_saliency(input_path, output_path, saliency_func):
  """Gets the saliency map and save it to visualization file.

  Args:
    input_path: path to the input jpeg file.
    output_path: path to the output jpeg file.
    saliency_func: a callable that takes [height, width, 3] RGB numpy image as
      input and outputs the [height, width] float array denoting saliency.
  """
  image_data = cv2.imread(input_path)[:, :, ::-1]  # To RGB.
  saliency = saliency_func(image_data)

  min_v, max_v = saliency.min(), saliency.max()
  saliency = (saliency- min_v) / (_SMALL_NUMBER + max_v - min_v)

  # Merge image and heatmap.
  heatmap = plotlib._py_convert_to_heatmap(saliency, normalize=False)

  saliency = np.expand_dims(saliency, -1)
  output = np.add(
      np.multiply(1.0 - saliency, image_data.astype(np.float32)),
      np.multiply(saliency, heatmap * 255.0))

  output = output.astype(np.uint8)
  cv2.imwrite(output_path, output[:, :, ::-1])  # To BGR.


def main(_):
  pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)
  tf.logging.info("Pipeline configure: %s", '=' * 128)
  tf.logging.info(pipeline_proto)

  g = tf.Graph()
  with g.as_default():

    # Infer saliency.

    image = tf.placeholder(tf.uint8, shape=[None, None, 3])
    image_resized = tf.image.resize_images(
        image, [pipeline_proto.eval_reader.image_height,
        pipeline_proto.eval_reader.image_width])

    model = builder.build(pipeline_proto.model, is_training=False)
    predictions = model.build_prediction(
        examples={ InputDataFields.image: tf.expand_dims(image_resized, 0)}, 
        prediction_task=GAPPredictionTasks.image_saliency)

    height, width = tf.shape(image)[0], tf.shape(image)[1]
    saliency = tf.image.resize_images(
        tf.expand_dims(predictions[GAPPredictions.image_saliency], axis=-1), 
        [height, width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)[:, :, :, 0]

    saver = tf.train.Saver()
    invalid_variable_names = tf.report_uninitialized_variables()

  with tf.Session(graph=g) as sess:

    # Load the latest checkpoint.

    checkpoint_path = tf.train.latest_checkpoint(pipeline_proto.model_dir)
    assert checkpoint_path is not None

    saver.restore(sess, checkpoint_path)
    assert len(sess.run(invalid_variable_names)) == 0

    # Iterate the testdir to generate the demo results.

    saliency_func = lambda x: sess.run(saliency[0], feed_dict={ image: x })

    for filename in os.listdir(FLAGS.image_path):
      tf.logging.info('On processing %s', filename)

      _get_saliency(
          input_path=os.path.join(FLAGS.image_path, filename),
          output_path=os.path.join(FLAGS.demo_path, filename),
          saliency_func=saliency_func)

  tf.logging.info('Done')

if __name__ == '__main__':
  tf.app.run()
