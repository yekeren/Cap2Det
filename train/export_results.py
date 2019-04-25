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
from core.training_utils import get_best_model_checkpoint
from protos import pipeline_pb2
from train import trainer

from tensorflow.python.platform import tf_logging as logging

flags = tf.app.flags

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string('pipeline_proto', '', 'Path to the pipeline proto file.')

flags.DEFINE_string('model_dir', '',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('model_path', '',
                    'Path to the checkpoints.')

flags.DEFINE_string('saved_ckpts_dir', '',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('detection_results_dir', '',
                    'Path to the directory saving results.')

flags.DEFINE_string('shard_indicator', '', '')

flags.DEFINE_string('input_pattern', '', '')

flags.DEFINE_integer('oicr_iterations', 3, '')

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


def _run_inference(pipeline_proto, checkpoint_path, oicr_iterations):
  """Runs the prediction.

  Args:
    pipeline_proto: An instance of pipeline_pb2.Pipeline.
    checkpoint_path: Path to the checkpoint file.
    oicr_iterations: A list of object_detection_evaluation.DetectionEvaluator.
  """
  eval_count = 0

  for examples in trainer.predict(pipeline_proto, checkpoint_path):
    batch_size = len(examples[InputDataFields.image_id])

    if eval_count == 0:
      class_labels = [
          x.decode('ascii')
          for x in examples[DetectionResultFields.class_labels]
      ]

    for i in range(batch_size):
      (image_id, image_height, image_width, num_groundtruths, groundtruth_boxes,
       groundtruth_classes) = (examples[InputDataFields.image_id][i],
                               examples[InputDataFields.image_height][i],
                               examples[InputDataFields.image_width][i],
                               examples[InputDataFields.num_objects][i],
                               examples[InputDataFields.object_boxes][i],
                               examples[InputDataFields.object_texts][i])

      oicr_iter = oicr_iterations
      num_detections, detection_boxes, detection_scores, detection_classes = (
          examples[DetectionResultFields.num_detections +
                   '_at_{}'.format(oicr_iter)][i],
          examples[DetectionResultFields.detection_boxes +
                   '_at_{}'.format(oicr_iter)][i],
          examples[DetectionResultFields.detection_scores +
                   '_at_{}'.format(oicr_iter)][i],
          examples[DetectionResultFields.detection_classes +
                   '_at_{}'.format(oicr_iter)][i])

      eval_count += 1
      if eval_count % 50 == 0:
        tf.logging.info('On image %i.', eval_count)

      # Write to detection result file.
      image_id = int(image_id.decode('ascii').split('.')[0])
      results = {
          'image_id': image_id,
          'bounding_boxes': []
      }

      for i in range(num_detections):
        ymin, xmin, ymax, xmax = [round(float(x), 3) for x in detection_boxes[i]]
        class_label = class_labels[int(detection_classes[i] - 1)]
        class_score = round(float(detection_scores[i]), 3)
        results['bounding_boxes'].append({
            'class_score': class_score,
            'class_label': class_label,
            'bounding_box': {
                'ymin': ymin,
                'xmin': xmin,
                'ymax': ymax,
                'xmax': xmax
            },
        })

      filename = os.path.join(FLAGS.detection_results_dir,
                              '{}.json'.format(image_id))
      with open(filename, 'w') as fid:
        fid.write(json.dumps(results, indent=2))
      tf.logging.info('image_id=%s, file=%s', image_id, filename)


def main(_):
  pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)

  if FLAGS.model_dir:
    pipeline_proto.model_dir = FLAGS.model_dir
    tf.logging.info("Override model checkpoint dir: %s", FLAGS.model_dir)

  if FLAGS.shard_indicator:
    pipeline_proto.eval_reader.wsod_reader.shard_indicator = FLAGS.shard_indicator
    tf.logging.info("Override shard_indicator: %s", FLAGS.shard_indicator)

  if FLAGS.input_pattern:
    while len(pipeline_proto.eval_reader.wsod_reader.input_pattern) > 0:
      pipeline_proto.eval_reader.wsod_reader.input_pattern.pop()
    pipeline_proto.eval_reader.wsod_reader.input_pattern.append(FLAGS.input_pattern)
    tf.logging.info("Override input_pattern: %s", FLAGS.input_pattern)

  tf.logging.info("Pipeline configure: %s", '=' * 128)
  tf.logging.info(pipeline_proto)

  checkpoint_path = FLAGS.model_path
  tf.logging.info('Start to evaluate checkpoint %s.', checkpoint_path)

  _run_inference(pipeline_proto, checkpoint_path, FLAGS.oicr_iterations)

  tf.logging.info('Done')


if __name__ == '__main__':
  tf.app.run()
