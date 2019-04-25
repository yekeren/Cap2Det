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
from core.standard_fields import NODPredictions
from core.standard_fields import NOD2Predictions
from core.training_utils import save_model_if_it_is_better
from core.training_utils import get_best_model_checkpoint
from core import plotlib
from protos import pipeline_pb2
from protos import mil_model_pb2
from protos import oicr_model_pb2
from protos import oicr_dilated_model_pb2
from protos import multi_resol_model_pb2
from protos import frcnn_model_pb2
from protos import wsod_voc_model_pb2
from protos import nod_model_pb2
from protos import nod2_model_pb2
from protos import nod3_model_pb2
from protos import nod4_model_pb2
from protos import stacked_attn_model_pb2
from train import trainer
from core.plotlib import _py_draw_rectangles
from core import box_utils

from object_detection.utils import object_detection_evaluation
from object_detection.metrics import coco_evaluation

from tensorflow.python.platform import tf_logging as logging

#from json import encoder
#
#encoder.FLOAT_REPR = lambda o: format(o, '.3f')

flags = tf.app.flags

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string('pipeline_proto', '', 'Path to the pipeline proto file.')

flags.DEFINE_string('model_dir', '',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('saved_ckpts_dir', '',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('vocabulary_file', '',
                    'Path to the detection vocabulary file.')

flags.DEFINE_string('detection_results_dir', '',
                    'Path to the directory saving results.')

flags.DEFINE_boolean('eval_coco_on_voc', False, '')

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


coco_to_voc = {
    5: 1,
    2: 2,
    15: 3,
    9: 4,
    40: 5,
    6: 6,
    3: 7,
    16: 8,
    57: 9,
    20: 10,
    61: 11,
    17: 12,
    18: 13,
    4: 14,
    1: 15,
    59: 16,
    19: 17,
    58: 18,
    7: 19,
    63: 20,
}


def _convert_coco_result_to_voc(boxes, scores, classes):
  """Directly converts coco detection results to voc detection results.

  Args:
    boxes: [num_boxes, 4] numpy float array.
    scores: [num_boxes] numpy float array.
    classes: [num_boxes] numpy string array.

  Returns:
    boxes: [num_boxes, 4] numpy float array.
    scores: [num_boxes] numpy float array.
    classes: [num_boxes] numpy string array.
  """
  det_boxes, det_scores, det_classes = [], [], []
  for box, score, cls in zip(boxes, scores, classes):
    if int(cls) in coco_to_voc:
      det_boxes.append(box)
      det_scores.append(score)
      det_classes.append(coco_to_voc[int(cls)])
  return np.stack(det_boxes, 0), np.stack(det_scores, 0), np.stack(
      det_classes, 0)


def _run_evaluation(pipeline_proto,
                    checkpoint_path,
                    oicr_iterations,
                    category_to_id,
                    categories,
                    save_report_to_file=False):
  """Runs the prediction.

  Args:
    pipeline_proto: An instance of pipeline_pb2.Pipeline.
    checkpoint_path: Path to the checkpoint file.
    oicr_iterations: A list of object_detection_evaluation.DetectionEvaluator.
    category_to_id: A python dict maps from the category name to integer id.
  """
  eval_count = 0

  for examples in trainer.predict(pipeline_proto, checkpoint_path):
    batch_size = len(examples[InputDataFields.image_id])

    if eval_count == 0:
      class_labels = [
          x.decode('utf8') for x in examples[DetectionResultFields.class_labels]
      ]

    for i in range(batch_size):
      (image_id, image_height, image_width, num_groundtruths, groundtruth_boxes,
       groundtruth_classes) = (examples[InputDataFields.image_id][i],
                               examples[InputDataFields.image_height][i],
                               examples[InputDataFields.image_width][i],
                               examples[InputDataFields.num_objects][i],
                               examples[InputDataFields.object_boxes][i],
                               examples[InputDataFields.object_texts][i])

      # Evaluate each OICR iterations.

      for oicr_iter in range(1 + oicr_iterations):
        num_detections, detection_boxes, detection_scores, detection_classes = (
            examples[DetectionResultFields.num_detections +
                     '_at_{}'.format(oicr_iter)][i],
            examples[DetectionResultFields.detection_boxes +
                     '_at_{}'.format(oicr_iter)][i],
            examples[DetectionResultFields.detection_scores +
                     '_at_{}'.format(oicr_iter)][i],
            examples[DetectionResultFields.detection_classes +
                     '_at_{}'.format(oicr_iter)][i])
        if FLAGS.eval_coco_on_voc:
          det_boxes, det_scores, det_classes = _convert_coco_result_to_voc(
              box_utils.py_coord_norm_to_abs(detection_boxes[:num_detections],
                                             image_height, image_width),
              detection_scores[:num_detections],
              detection_classes[:num_detections])

      eval_count += 1
      if eval_count % 50 == 0:
        tf.logging.info('On image %i.', eval_count)

      # Write to detection result file.

      if FLAGS.detection_results_dir:
        results = []
        detection_boxes = box_utils.py_coord_norm_to_abs(
            detection_boxes[:num_detections], image_height, image_width)

        image_id = image_id.decode('utf8').split('.')[0]
        for i in range(num_detections):
          ymin, xmin, ymax, xmax = detection_boxes[i]
          ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
          category_id = class_labels[int(detection_classes[i] - 1)]
          results.append({
              'image_id': image_id,
              'category_id': category_id,
              'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
              'score': round(float(detection_scores[i]), 5),
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
    pipeline_proto.eval_reader.shard_indicator = FLAGS.shard_indicator
    tf.logging.info("Override shard_indicator: %s", FLAGS.shard_indicator)

  if FLAGS.input_pattern:
    while len(pipeline_proto.eval_reader.wsod_reader.input_pattern) > 0:
      pipeline_proto.eval_reader.wsod_reader.input_pattern.pop()
    pipeline_proto.eval_reader.wsod_reader.input_pattern.append(FLAGS.input_pattern)
    tf.logging.info("Override input_pattern: %s", FLAGS.input_pattern)

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

  #checkpoint_path = get_best_model_checkpoint(FLAGS.saved_ckpts_dir)

  checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
  tf.logging.info('Start to evaluate checkpoint %s.', checkpoint_path)

  _run_evaluation(
      pipeline_proto,
      checkpoint_path,
      FLAGS.oicr_iterations,
      category_to_id,
      categories,
      save_report_to_file=True)

  tf.logging.info('Done')


if __name__ == '__main__':
  tf.app.run()
