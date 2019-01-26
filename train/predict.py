from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json
import numpy as np
import collections
import tensorflow as tf
from google.protobuf import text_format

from core.standard_fields import InputDataFields
from core.standard_fields import DetectionResultFields
from protos import pipeline_pb2
from protos import mil_model_pb2
from protos import oicr_model_pb2
from protos import oicr_dilated_model_pb2
from protos import multi_resol_model_pb2
from protos import frcnn_model_pb2
from train import trainer
from core.plotlib import _py_draw_rectangles

from object_detection.utils import object_detection_evaluation

flags = tf.app.flags

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string('evaluator', 'pascal',
                    "Name of the evaluator, can either be `coco` or `pascal`.")

flags.DEFINE_string('pipeline_proto', '', 'Path to the pipeline proto file.')

flags.DEFINE_string('model_dir', '',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('eval_log_dir', '',
                    'Path to the directory saving eval logs.')

flags.DEFINE_string('vocabulary_file', '',
                    'Path to the detection vocabulary file.')

flags.DEFINE_integer('eval_steps', 500, 'Number of steps to evaluate.')

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


def _run_evaluation(pipeline_proto, checkpoint_path, evaluators,
                    category_to_id):
  """Runs the prediction.

  Args:
    pipeline_proto: An instance of pipeline_pb2.Pipeline.
    checkpoint_path: Path to the checkpoint file.
    evaluators: A list of object_detection_evaluation.DetectionEvaluator.
    category_to_id: A python dict maps from the category name to integer id.
  """
  count = 0
  for examples in trainer.predict(pipeline_proto, checkpoint_path):
    batch_size = len(examples[InputDataFields.image_id])
    summary_bytes = examples['summary']

    if count == 0:
      summary = tf.Summary().FromString(summary_bytes)

    for i in range(batch_size):
      (image_id, num_groundtruths, groundtruth_boxes,
       groundtruth_classes) = (examples[InputDataFields.image_id][i],
                               examples[InputDataFields.num_objects][i],
                               examples[InputDataFields.object_boxes][i],
                               examples[InputDataFields.object_texts][i])

      for oicr_iter, evaluator in enumerate(evaluators):

        num_detections, detection_boxes, detection_scores, detection_classes = (
            examples[DetectionResultFields.num_detections +
                     '_at_{}'.format(oicr_iter)][i],
            examples[DetectionResultFields.detection_boxes +
                     '_at_{}'.format(oicr_iter)][i],
            examples[DetectionResultFields.detection_scores +
                     '_at_{}'.format(oicr_iter)][i],
            examples[DetectionResultFields.detection_classes +
                     '_at_{}'.format(oicr_iter)][i])

        # Add ground-truth annotations.

        evaluator.add_single_ground_truth_image_info(
            image_id, {
                'groundtruth_boxes':
                groundtruth_boxes[:num_groundtruths],
                'groundtruth_classes':
                np.array([
                    category_to_id[x.decode('utf8')]
                    for x in groundtruth_classes[:num_groundtruths]
                ]),
                'groundtruth_difficult':
                np.zeros([num_groundtruths], dtype=np.bool)
            })

        # Add detection results.

        evaluator.add_single_detected_image_info(
            image_id, {
                'detection_boxes': detection_boxes[:num_detections],
                'detection_scores': detection_scores[:num_detections],
                'detection_classes': detection_classes[:num_detections]
            })

      count += 1
      if count % 50 == 0:
        tf.logging.info('On image %i.', count)
    if count > FLAGS.eval_steps:
      break

  for oicr_iter, evaluator in enumerate(evaluators):
    metrics = evaluator.evaluate()
    evaluator.clear()
    tf.logging.info('\n%s', json.dumps(metrics, indent=2))

    for k, v in metrics.items():
      summary.value.add(tag='{}_iter{}'.format(k, oicr_iter), simple_value=v)

  return summary


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

  # Create the evaluator.

  number_of_evaluators = 0
  if pipeline_proto.model.HasExtension(mil_model_pb2.MILModel.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        mil_model_pb2.MILModel.ext].oicr_iterations
  if pipeline_proto.model.HasExtension(oicr_model_pb2.OICRModel.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        oicr_model_pb2.OICRModel.ext].oicr_iterations
  if pipeline_proto.model.HasExtension(oicr_dilated_model_pb2.OICRDilatedModel.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        oicr_dilated_model_pb2.OICRDilatedModel.ext].oicr_iterations
  if pipeline_proto.model.HasExtension(multi_resol_model_pb2.MultiResolModel.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        multi_resol_model_pb2.MultiResolModel.ext].oicr_iterations
  if pipeline_proto.model.HasExtension(frcnn_model_pb2.FRCNNModel.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        frcnn_model_pb2.FRCNNModel.ext].oicr_iterations

  number_of_evaluators = max(1, number_of_evaluators) 

  if FLAGS.evaluator.lower() == 'pascal':
    evaluators = [
        object_detection_evaluation.PascalDetectionEvaluator(categories)
        for i in range(number_of_evaluators)
    ]
  else:
    raise ValueError('Invalid evaluator {}.'.format(FLAGS.evaluator))

  # Evaluation loop.

  latest_step = None
  while True:
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    if checkpoint_path is not None:
      global_step = int(checkpoint_path.split('-')[-1])

      if global_step != latest_step and global_step > 400:

        # Evaluate the checkpoint.

        latest_step = global_step
        tf.logging.info('Start to evaluate checkpoint %s.', checkpoint_path)

        summary = _run_evaluation(pipeline_proto, checkpoint_path, evaluators,
                                  category_to_id)

        # Write summary.
        summary_writer = tf.summary.FileWriter(FLAGS.eval_log_dir)
        summary_writer.add_summary(summary, global_step=global_step)
        summary_writer.close()
        tf.logging.info("Summary is written.")
        continue
    tf.logging.info("Wait for 10 seconds.")
    time.sleep(10)

  evaluator.clear()

  tf.logging.info('Done')


if __name__ == '__main__':
  tf.app.run()
