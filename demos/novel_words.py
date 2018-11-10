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

flags.DEFINE_string('pipeline_proto', '', 'Path to the pipeline proto file.')

flags.DEFINE_string('name_to_class_id_file', '',
                    'Path to the name_to_class_id file.')

flags.DEFINE_string('model_dir', '',
                    'Path to the directory storing model checkpoints.')

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


def _knn_retrieval(queries, vocabulary, word_embedding, word_saliency):
  """Processes kNN retrieval to search the synonyms.

  Args:
    queries: a list of strings denoting the queries.
    vocabulary: words in the vocabulary, excluding `UNK` symbol.
    word_embedding: a numpy array of shape [len(vocabulary), dims].
    word_saliency: a numpy array of shape [len(vocabulary)].

  Returns:
    synonyms_list: a list of length(queries), in which each element is a list of
      synonym tuple(word, similarity, saliency).
  """
  vocabulary = [word.decode('UTF8') for word in vocabulary.tolist()]
  indices = [vocabulary.index(word) for word in queries]

  # `query_embedding` shape = [len(queries), dims].

  query_embedding = word_embedding[indices]
  similarity = np.matmul(word_embedding, query_embedding.transpose())

  # kNN retrieval.

  synonyms_list = [[] for query in queries]
  for i, (word, similarity_row) in enumerate(zip(vocabulary, similarity)):
    nearest_query = similarity_row.argmax()
    synonyms_list[nearest_query].append({
        'word':
        word,
        'similarity':
        similarity_row[nearest_query],
        'saliency':
        word_saliency[i]
    })

  # Sort by similarity.
  for i in range(len(synonyms_list)):
    synonyms_list[i] = sorted(
        synonyms_list[i], key=lambda x: x['similarity'], reverse=True)
  return synonyms_list


def main(_):
  pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)

  if FLAGS.model_dir:
    pipeline_proto.model_dir = FLAGS.model_dir
    tf.logging.info("Override model checkpoint dir: %s", FLAGS.model_dir)

  tf.logging.info("Pipeline configure: %s", '=' * 128)
  tf.logging.info(pipeline_proto)

  g = tf.Graph()
  with g.as_default():
    model = builder.build(pipeline_proto.model, is_training=False)
    predictions = model.build_prediction(
        examples={}, prediction_task=GAPPredictionTasks.word_saliency)
    saver = tf.train.Saver()
    invalid_variable_names = tf.report_uninitialized_variables()

  with tf.Session(graph=g) as sess:
    sess.run(tf.tables_initializer())

    checkpoint_path = tf.train.latest_checkpoint(pipeline_proto.model_dir)
    assert checkpoint_path is not None

    saver.restore(sess, checkpoint_path)
    assert len(sess.run(invalid_variable_names)) == 0

    predictions = sess.run(predictions)

  (vocabulary, word_saliency,
   word_embedding) = (predictions[GAPPredictions.vocabulary],
                      predictions[GAPPredictions.word_saliency],
                      predictions[GAPPredictions.word_embedding])
  vocabulary = [word.decode('UTF8') for word in vocabulary.tolist()]

  # Read the MSCOCO vocabulary.

  name_to_class_id = {}
  with open(FLAGS.name_to_class_id_file, 'r') as fid:
    for line in fid.readlines():
      name, class_id = line.strip('\n').split('\t')
      name_to_class_id[name] = class_id
  queries = list(name_to_class_id)
  query_embedding = word_embedding[[
      vocabulary.index(query) for query in queries
  ]]

  # Rank the words by importance.

  similarity_to_coco = np.matmul(word_embedding, query_embedding.transpose())
  most_similar_coco_word_index = similarity_to_coco.argmax(axis=-1)

  indices = np.argsort(word_saliency)[::-1]
  for i in indices[:1000]:
    word, score = vocabulary[i], word_saliency[i]
    coco_i = most_similar_coco_word_index[i]
    print('%s,%.4lf,%s,%.4lf' % (word, score, queries[coco_i],
                                 similarity_to_coco[i][coco_i]))

  tf.logging.info('Done')


if __name__ == '__main__':
  tf.app.run()
