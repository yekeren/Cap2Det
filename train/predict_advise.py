from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np
import collections
import tensorflow as tf
from google.protobuf import text_format

from protos import pipeline_pb2
from train import trainer

flags = tf.app.flags

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string('pipeline_proto', '', 'Path to the pipeline proto file.')

flags.DEFINE_string('model_dir', '',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('qa_json_path', 'raw_data/qa.json/',
                    'Path to the qa annotation.')

flags.DEFINE_string('prediction_output_path', 'results.json',
                    'Path to the prediction results.')

flags.DEFINE_string('input_pattern', '',
                    'Path to the prediction results.')

flags.DEFINE_string('metrics_output_format', 'csv',
                    'Format to output the metrics.')

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


def _revise_image_id(image_id):
  """Revises image id.

  Args:
    image_id: Image ID in numeric number format.

  Returns:
    Image ID in `subdir/filename` format.
  """
  if image_id >= 170000:
    image_id = '10/{}.png'.format(image_id)
  else:
    image_id = '{}/{}.jpg'.format(image_id % 10, image_id)
  return image_id


def _load_json(filename):
  """Loads data in json format.

  Args:
    filename: Path to the json file.

  Returns:
    a python dict representing the json object.
  """
  with open(filename, 'r') as fid:
    return json.load(fid)


def _update_metrics(metrics, groundtruth_list, prediction_list):
  """Updtes the metrics.

  Args:
    metrics: An instance of collections.defaultdict.
    groundtruth_list: A list of strings, which are the groundtruth annotations.
    prediction_list: A list of predictions.

  Returns:
    updated metrics.
  """
  if not groundtruth_list:
    return metrics

  bingo = 1.0 if prediction_list[0] in groundtruth_list else 0.0
  ranks = [1.0 + prediction_list.index(x) for x in groundtruth_list]

  metrics['Accuracy'].append(bingo)
  metrics['RankMin'].append(np.min(ranks))
  metrics['RankAvg'].append(np.mean(ranks))
  metrics['RankMed'].append(np.median(ranks))
  return metrics


def _summerize_metrics(metrics):
  if not 'Accuracy' in metrics:
    return

  accuracy = round(np.mean(metrics['Accuracy']), 3)
  rank_min = round(np.mean(metrics['RankMin']), 3)
  rank_avg = round(np.mean(metrics['RankAvg']), 3)
  rank_med = round(np.mean(metrics['RankMed']), 3)

  if FLAGS.metrics_output_format == 'csv':
    print(',Accuracy,RankMin,RankAvg,RankMed')
    print('{},{},{},{},{}'.format(FLAGS.pipeline_proto, accuracy, rank_min,
                                  rank_avg, rank_med))

  elif FLAGS.metrics_output_format == 'html':
    print(
        '<tr><th></th><th>Accuracy</th><th>RankMin</th><th>RankAvg</th><th>RankMed</th></tr>'
    )
    print('<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.
          format(FLAGS.pipeline_proto, accuracy, rank_min, rank_avg, rank_med))

  else:
    raise ValueError('Invalid output format {}'.format(
        FLAGS.metrics_output_format))


def _run_prediction(pipeline_proto):
  """Runs the prediction.

  Args:
    pipeline_proto: an instance of pipeline_pb2.Pipeline.
  """
  results = {}
  metrics = collections.defaultdict(list)

  for example_index, example in enumerate(trainer.predict(pipeline_proto)):

    # Compute the metrics.
    image_id = example['image_id'][0]

    annotation = _load_json(
        os.path.join(FLAGS.qa_json_path, '{}.json'.format(image_id)))
    (groundtruth_list, question_list) = (annotation['groundtruth_list'],
                                         annotation['question_list'])
    prediction_list = [
        question_list[i] for i in example['similarity'][0].argsort()[::-1]
    ]
    _update_metrics(metrics, groundtruth_list, prediction_list)

    # Create the result entry to write into the .json file.

    results[_revise_image_id(image_id)] = [
        question_list[index]
        for index in np.argsort(example['similarity'][0])[::-1]
    ]

    if example_index % 100 == 0:
      tf.logging.info('On image %i', example_index)

  with open(FLAGS.prediction_output_path, 'w') as fid:
    fid.write(json.dumps(results, indent=2))

  _summerize_metrics(metrics)


def main(_):
  pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)

  if FLAGS.model_dir:
    pipeline_proto.model_dir = FLAGS.model_dir
    tf.logging.info("Override model checkpoint dir: %s", FLAGS.model_dir)

  if FLAGS.input_pattern:
    while len(pipeline_proto.eval_reader.advise_reader.input_pattern) > 0:
      pipeline_proto.eval_reader.advise_reader.input_pattern.pop()
    pipeline_proto.eval_reader.advise_reader.input_pattern.append(FLAGS.input_pattern)
    tf.logging.info("Override model input_pattern: %s", FLAGS.input_pattern)

  tf.logging.info("Pipeline configure: %s", '=' * 128)
  tf.logging.info(pipeline_proto)

  _run_prediction(pipeline_proto)

  tf.logging.info('Done')


if __name__ == '__main__':
  tf.app.run()
