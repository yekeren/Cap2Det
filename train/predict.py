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

flags.DEFINE_string('evaluator', 'pascal',
                    "Name of the evaluator, can either be `coco` or `pascal`.")

flags.DEFINE_string('pipeline_proto', '', 'Path to the pipeline proto file.')

flags.DEFINE_string('model_dir', '',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('saved_ckpts_dir', '',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('eval_log_dir', '',
                    'Path to the directory saving eval logs.')

flags.DEFINE_string('vocabulary_file', '',
                    'Path to the detection vocabulary file.')

flags.DEFINE_integer('max_eval_examples', 500,
                     'Number of examples to evaluate.')

flags.DEFINE_integer('max_visl_examples', 100, 'Minimum eval steps.')

flags.DEFINE_integer('min_eval_steps', 2000, 'Minimum eval steps.')

flags.DEFINE_string('results_dir', 'results',
                    'Path to the directory saving results.')

flags.DEFINE_string('detection_result_dir', '',
                    'Path to the directory saving results.')

flags.DEFINE_integer('visl_size', 500, '')
flags.DEFINE_string('visl_file_path', '', '')

flags.DEFINE_float('min_visl_detection_score', 0.01, '')

flags.DEFINE_boolean('run_once', False, '')
flags.DEFINE_boolean('eval_coco_on_voc', False, '')

flags.DEFINE_string('shard_indicator', '', '')

flags.DEFINE_string('input_pattern', '', '')

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


def _visualize(examples, categories, filename):
  """Visualizes exmaples.

  Args:
    examples: A list of python dict saving examples to be visualized.
    filename: Path to the output file.
  """
  with open(filename, 'w') as fid:
    fid.write('<table border=1>')
    for example_index, example in enumerate(examples):
      (image_id, image, image_height, image_width, num_gt_boxes, gt_boxes,
       gt_labels, num_dt_boxes, dt_boxes, dt_scores,
       dt_labels) = (example[InputDataFields.image_id],
                     example[InputDataFields.image],
                     example[InputDataFields.image_height],
                     example[InputDataFields.image_width],
                     example[InputDataFields.num_objects],
                     example[InputDataFields.object_boxes],
                     example[InputDataFields.object_texts],
                     example[DetectionResultFields.num_detections],
                     example[DetectionResultFields.detection_boxes],
                     example[DetectionResultFields.detection_scores],
                     example[DetectionResultFields.detection_classes])

      # Print captions.

      caption_annot = ''
      if (InputDataFields.num_captions in example and
          InputDataFields.caption_strings in example and
          InputDataFields.caption_lengths in example):
        (num_captions, caption_strings,
         caption_lengths) = (example[InputDataFields.num_captions],
                             example[InputDataFields.caption_strings],
                             example[InputDataFields.caption_lengths])
        captions = []
        for caption_string, caption_length in zip(
            caption_strings[:num_captions], caption_lengths[:num_captions]):
          captions.append(' '.join(
              [x.decode('utf8') for x in caption_string[:caption_length]]))
        caption_annot = '</br>'.join(captions)

      # Generated image-level ground-truth.

      labels_gt_annot = ''
      labels_ps_annot = ''
      if 'debug_groundtruth_labels' in example and 'debug_pseudo_labels' in example:
        labels_gt = [
            categories[i]
            for i, v in enumerate(example['debug_groundtruth_labels'])
            if v > 0
        ]
        labels_ps = [
            categories[i]
            for i, v in enumerate(example['debug_pseudo_labels'])
            if v > 0
        ]
        labels_gt_annot = ','.join(labels_gt)
        labels_ps_annot = ','.join(labels_ps)
        if labels_ps_annot:
          labels_ps_annot = 'pseudo:' + labels_ps_annot

      # Image canvas.

      image = cv2.resize(image, (image_width, image_height))

      # Image with ground-truth boxes.

      image_with_gt = plotlib._py_draw_rectangles_v2(
          image,
          num_gt_boxes,
          gt_boxes,
          np.zeros_like(gt_labels, dtype=np.float32),
          gt_labels,
          color=plotlib.BLUE,
          show_score=False)
      image_with_gt = cv2.cvtColor(image_with_gt, cv2.COLOR_RGB2BGR)
      gt_base64 = plotlib._py_convert_to_base64(image_with_gt)

      # Image with predicted boxes.

      for i, dt_score in enumerate(dt_scores):
        if dt_score < FLAGS.min_visl_detection_score:
          break

      num_dt_boxes = min(i, num_dt_boxes)
      dt_labels = np.array(
          [categories[int(x) - 1].encode('utf8') for x in dt_labels])

      recall_mask, precision_mask = box_utils.py_evaluate_precision_and_recall(
          num_gt_boxes, gt_boxes, gt_labels, num_dt_boxes, dt_boxes, dt_labels)
      image_with_dt = plotlib._py_draw_rectangles_v2(
          image,
          num_dt_boxes,
          dt_boxes,
          dt_scores,
          dt_labels,
          color=plotlib.RED,
          show_score=True)
      image_with_dt = plotlib._py_draw_rectangles_v2(
          image_with_dt,
          precision_mask.astype(np.int32).sum(),
          dt_boxes[precision_mask],
          dt_scores[precision_mask],
          dt_labels[precision_mask],
          color=plotlib.GREEN,
          show_score=True)
      image_with_dt = cv2.cvtColor(image_with_dt, cv2.COLOR_RGB2BGR)
      dt_base64 = plotlib._py_convert_to_base64(image_with_dt)

      # Write html file.

      fid.write('<tr>')
      fid.write(
          '<td><img src="data:image/jpg;base64,%s"></br>%s</br>%s</br>%s</td>' %
          (gt_base64, caption_annot, labels_gt_annot, labels_ps_annot))
      fid.write('<td><img src="data:image/jpg;base64,%s"></td>' % (dt_base64))
      fid.write('</tr>')
    fid.write('</table>')
  tf.logging.info('File is written to %s, #images=%i', filename, example_index)


def _analyze_latent_variables(proba_h_given_c, categories):
  """Analyzes latent varibles.

  Args:
    proba_h_given_c: A [num_latent_factors, num_classes] numpy array.
  """
  for (data, category) in zip(proba_h_given_c.transpose(), categories):
    factors = []
    for i in data.argsort()[::-1]:
      if data[i] <= 0.05: break
      factors.append('%i:%.3lf' % (i, data[i]))
    tf.logging.info('Class %s: %s', category['name'], ', '.join(factors))


def _analyze_tokens(tokens, per_class_token_scores):
  top_k = 10
  num_classes = per_class_token_scores.shape[1]
  for i in range(num_classes):
    elems = []
    indices = np.argsort(per_class_token_scores[:, i])[::-1][:top_k]
    for j in indices:
      score = per_class_token_scores[j][i]
      elems.append('%s:%.2lf' % (tokens[j].decode('utf8'), score))

    tf.logging.info('%i ---- %s', i, ','.join(elems))


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
                    evaluators,
                    category_to_id,
                    categories,
                    save_report_to_file=False):
  """Runs the prediction.

  Args:
    pipeline_proto: An instance of pipeline_pb2.Pipeline.
    checkpoint_path: Path to the checkpoint file.
    evaluators: A list of object_detection_evaluation.DetectionEvaluator.
    category_to_id: A python dict maps from the category name to integer id.
  """
  eval_count = 0
  visl_examples = []

  for examples in trainer.predict(pipeline_proto, checkpoint_path):
    batch_size = len(examples[InputDataFields.image_id])
    summary_bytes = examples['summary']

    if eval_count == 0:
      summary = tf.Summary().FromString(summary_bytes)
      if NODPredictions.midn_proba_h_given_c in examples:
        _analyze_latent_variables(examples[NODPredictions.midn_proba_h_given_c],
                                  categories)
      if 'tokens' in examples and 'per_class_token_scores' in examples:
        _analyze_tokens(examples['tokens'], examples['per_class_token_scores'])

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
        evaluator.add_single_ground_truth_image_info(
            image_id, {
                'groundtruth_boxes':
                box_utils.py_coord_norm_to_abs(
                    groundtruth_boxes[:num_groundtruths], image_height,
                    image_width),
                'groundtruth_classes':
                np.array([
                    category_to_id[x.decode('utf8')]
                    for x in groundtruth_classes[:num_groundtruths]
                ]),
                'groundtruth_difficult':
                np.zeros([num_groundtruths], dtype=np.bool)
            })
        if not FLAGS.eval_coco_on_voc:
          evaluator.add_single_detected_image_info(
              image_id, {
                  'detection_boxes':
                  box_utils.py_coord_norm_to_abs(
                      detection_boxes[:num_detections], image_height,
                      image_width),
                  'detection_scores':
                  detection_scores[:num_detections],
                  'detection_classes':
                  detection_classes[:num_detections]
              })
        else:
          det_boxes, det_scores, det_classes = _convert_coco_result_to_voc(
              box_utils.py_coord_norm_to_abs(detection_boxes[:num_detections],
                                             image_height, image_width),
              detection_scores[:num_detections],
              detection_classes[:num_detections])

          evaluator.add_single_detected_image_info(
              image_id, {
                  'detection_boxes': det_boxes,
                  'detection_scores': det_scores,
                  'detection_classes': det_classes
              })

      eval_count += 1
      if eval_count % 50 == 0:
        tf.logging.info('On image %i.', eval_count)

      # Add to visualization list.

      if len(visl_examples) < FLAGS.max_visl_examples:
        visl_example = {
            InputDataFields.image_id: examples[InputDataFields.image_id][i],
            InputDataFields.image: examples[InputDataFields.image][i],
            InputDataFields.image_height:
            examples[InputDataFields.image_height][i],
            InputDataFields.image_width:
            examples[InputDataFields.image_width][i],
            InputDataFields.num_objects:
            examples[InputDataFields.num_objects][i],
            InputDataFields.object_boxes:
            examples[InputDataFields.object_boxes][i],
            InputDataFields.object_texts:
            examples[InputDataFields.object_texts][i],
            DetectionResultFields.num_detections: num_detections,
            DetectionResultFields.detection_boxes: detection_boxes,
            DetectionResultFields.detection_scores: detection_scores,
            DetectionResultFields.detection_classes: detection_classes
        }
        if InputDataFields.num_captions in examples:
          visl_example[InputDataFields.num_captions] = examples[
              InputDataFields.num_captions][i]
        if InputDataFields.caption_strings in examples:
          visl_example[InputDataFields.caption_strings] = examples[
              InputDataFields.caption_strings][i]
        if InputDataFields.caption_lengths in examples:
          visl_example[InputDataFields.caption_lengths] = examples[
              InputDataFields.caption_lengths][i]
        if 'debug_groundtruth_labels' in examples:
          visl_example['debug_groundtruth_labels'] = examples[
              'debug_groundtruth_labels'][i]
        if 'debug_pseudo_labels' in examples:
          visl_example['debug_pseudo_labels'] = examples['debug_pseudo_labels'][
              i]
        visl_examples.append(visl_example)

      # Write to detection result file.

      if FLAGS.detection_result_dir:
        results = []
        detection_boxes = box_utils.py_coord_norm_to_abs(
            detection_boxes[:num_detections], image_height, image_width)

        image_id = int(image_id.decode('utf8'))
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

        filename = os.path.join(FLAGS.detection_result_dir,
                                '{}.json'.format(image_id))
        with open(filename, 'w') as fid:
          fid.write(json.dumps(results, indent=2))
        tf.logging.info('image_id=%s, file=%s', image_id, filename)

    if eval_count > FLAGS.max_eval_examples:
      break

  # Visualize the results.

  if FLAGS.visl_file_path:
    _visualize(visl_examples, class_labels, FLAGS.visl_file_path)

  for oicr_iter, evaluator in enumerate(evaluators):
    metrics = evaluator.evaluate()
    evaluator.clear()
    for k, v in metrics.items():
      summary.value.add(tag='{}_iter{}'.format(k, oicr_iter), simple_value=v)
    tf.logging.info('\n%s', json.dumps(metrics, indent=2))

    # Write the result file.
    if save_report_to_file:
      if FLAGS.evaluator == 'pascal':
        corloc = [('/'.join(k.split('/')[1:]), v)
                  for k, v in metrics.items()
                  if 'CorLoc' in k]
        mAP = [('/'.join(k.split('/')[1:]), v)
               for k, v in metrics.items()
               if 'AP' in k]

        filename = os.path.join(FLAGS.results_dir,
                                FLAGS.pipeline_proto.split('/')[-1])
        filename = filename.replace('pbtxt',
                                    'csv') + 'iter_{}'.format(oicr_iter)
        with open(filename, 'w') as fid:
          fid.write('{}\n'.format(eval_count))
          fid.write('\n')
          for lst in [mAP, corloc]:
            line1 = ','.join([k for k, _ in lst]).replace(
                '@0.5IOU', '').replace('AP/', '').replace('CorLoc/', '')
            line2 = ' , '.join(['%.1lf' % (v * 100) for _, v in lst])

            fid.write(line1 + '\n')
            fid.write(line2 + '\n')
            fid.write('\n')
            fid.write(line1.replace(',', '&') + '\n')
            fid.write(line2.replace(',', '&') + '\n')
            fid.write('\n')
      #elif FLAGS.evaluator == 'coco':
      #  value = [('/'.join(k.split('/')[1:]), v) for k, v in metrics.items()]
      #  filename = os.path.join(FLAGS.results_dir,
      #                          FLAGS.pipeline_proto.split('/')[-1])
      #  filename = filename.replace('pbtxt', 'csv')
      #  with open(filename, 'w') as fid:
      #    fid.write('{}\n'.format(eval_count))
      #    fid.write('\n')
      #    line1 = ' , '.join([k for k, _ in value])
      #    line2 = ' , '.join(['%.1lf' % (v * 100) for _, v in value])

      #    fid.write(line1 + '\n')
      #    fid.write(line2 + '\n')
      #    fid.write('\n')
      #    fid.write(line1.replace(',', '&') + '\n')
      #    fid.write(line2.replace(',', '&') + '\n')
      #    fid.write('\n')

  if 'PascalBoxes_Precision/mAP@0.5IOU' in metrics:
    return summary, metrics['PascalBoxes_Precision/mAP@0.5IOU']
  return summary, metrics['DetectionBoxes_Precision/mAP']


def main(_):
  pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)

  if FLAGS.model_dir:
    pipeline_proto.model_dir = FLAGS.model_dir
    tf.logging.info("Override model checkpoint dir: %s", FLAGS.model_dir)

  if FLAGS.shard_indicator:
    pipeline_proto.eval_reader.shard_indicator = FLAGS.shard_indicator
    tf.logging.info("Override shard_indicator: %s", FLAGS.shard_indicator)

  if FLAGS.input_pattern:
    while len(pipeline_proto.eval_reader.input_pattern) > 0:
      pipeline_proto.eval_reader.input_pattern.pop()
    pipeline_proto.eval_reader.input_pattern.append(FLAGS.input_pattern)
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

  # Create the evaluator.

  number_of_evaluators = 0
  if pipeline_proto.model.HasExtension(mil_model_pb2.MILModel.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        mil_model_pb2.MILModel.ext].oicr_iterations
  if pipeline_proto.model.HasExtension(oicr_model_pb2.OICRModel.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        oicr_model_pb2.OICRModel.ext].oicr_iterations
  if pipeline_proto.model.HasExtension(
      oicr_dilated_model_pb2.OICRDilatedModel.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        oicr_dilated_model_pb2.OICRDilatedModel.ext].oicr_iterations
  if pipeline_proto.model.HasExtension(
      multi_resol_model_pb2.MultiResolModel.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        multi_resol_model_pb2.MultiResolModel.ext].oicr_iterations
  if pipeline_proto.model.HasExtension(frcnn_model_pb2.FRCNNModel.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        frcnn_model_pb2.FRCNNModel.ext].oicr_iterations
  if pipeline_proto.model.HasExtension(wsod_voc_model_pb2.WsodVocModel.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        wsod_voc_model_pb2.WsodVocModel.ext].oicr_iterations
  if pipeline_proto.model.HasExtension(nod_model_pb2.NODModel.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        nod_model_pb2.NODModel.ext].oicr_iterations
  if pipeline_proto.model.HasExtension(
      stacked_attn_model_pb2.StackedAttnModel.ext):
    number_of_evaluators = pipeline_proto.model.Extensions[
        stacked_attn_model_pb2.StackedAttnModel.ext].saddn_iterations
  if pipeline_proto.model.HasExtension(nod2_model_pb2.NOD2Model.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        nod2_model_pb2.NOD2Model.ext].oicr_iterations
  if pipeline_proto.model.HasExtension(nod3_model_pb2.NOD3Model.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        nod3_model_pb2.NOD3Model.ext].oicr_iterations
  if pipeline_proto.model.HasExtension(nod4_model_pb2.NOD4Model.ext):
    number_of_evaluators = 1 + pipeline_proto.model.Extensions[
        nod4_model_pb2.NOD4Model.ext].oicr_iterations

  number_of_evaluators = max(1, number_of_evaluators)

  if FLAGS.evaluator.lower() == 'pascal':
    evaluators = [
        object_detection_evaluation.PascalDetectionEvaluator(categories)
        for i in range(number_of_evaluators)
    ]
  elif FLAGS.evaluator.lower() == 'coco':
    evaluators = [
        coco_evaluation.CocoDetectionEvaluator(categories)
        for i in range(number_of_evaluators)
    ]
  else:
    raise ValueError('Invalid evaluator {}.'.format(FLAGS.evaluator))

  if not FLAGS.run_once:

    # Evaluation loop.

    latest_step = None
    while True:
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
      if checkpoint_path is not None:
        global_step = int(checkpoint_path.split('-')[-1])

        if global_step != latest_step and global_step >= FLAGS.min_eval_steps:

          # Evaluate the checkpoint.

          latest_step = global_step
          tf.logging.info('Start to evaluate checkpoint %s.', checkpoint_path)

          summary, metric = _run_evaluation(pipeline_proto, checkpoint_path,
                                            evaluators, category_to_id,
                                            categories)

          step_best, metric_best = save_model_if_it_is_better(
              global_step, metric, checkpoint_path, FLAGS.saved_ckpts_dir)

          # Write summary.
          summary.value.add(tag='loss/best_metric', simple_value=metric_best)
          summary_writer = tf.summary.FileWriter(FLAGS.eval_log_dir)
          summary_writer.add_summary(summary, global_step=global_step)
          summary_writer.close()
          tf.logging.info("Summary is written.")

          continue
      tf.logging.info("Wait for 10 seconds.")
      time.sleep(10)

  else:

    # Run once.
    #checkpoint_path = get_best_model_checkpoint(FLAGS.saved_ckpts_dir)
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    tf.logging.info('Start to evaluate checkpoint %s.', checkpoint_path)

    summary, metric = _run_evaluation(
        pipeline_proto,
        checkpoint_path,
        evaluators,
        category_to_id,
        categories,
        save_report_to_file=True)

  tf.logging.info('Done')


if __name__ == '__main__':
  tf.app.run()
