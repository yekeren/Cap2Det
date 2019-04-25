from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import cv2
import nltk

import numpy as np
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('tfrecord_output_path', 'data/train/train.record',
                    'Path to the directory saving output .tfrecord files.')

flags.DEFINE_integer('number_of_parts', 100, 'Number of output parts.')

flags.DEFINE_string('image_data_path', 'raw_data/train_images/',
                    'Path to the directory saving images in jpeg format.')

flags.DEFINE_string('img_feature_npy_path', 'raw_data/advise.img.npy/',
                    'Path to the directory saving img features in npy format.')

flags.DEFINE_string('roi_feature_npy_path', 'raw_data/wsod.roi.npy/',
                    'Path to the directory saving roi features in npy format.')

flags.DEFINE_string(
    'qa_json_path', 'raw_data/qa.json/',
    'Path to the directory saving qa annotations in json format.')

flags.DEFINE_string(
    'ocr_json_path', 'raw_data/ocr.json/',
    'Path to the directory saving ocr annotations in json format.')

flags.DEFINE_string(
    'densecap_json_path', 'raw_data/densecap.json/',
    'Path to the directory saving densecap annotations in json format.')

flags.DEFINE_string('symbol_npy_path', 'raw_data/symbol.npy/',
                    'Path to the directory saving roi features in npy format.')

FLAGS = flags.FLAGS

_max_ocr_text_len = 0
_max_densecap_text_len = 0


def _tokenize(sentence):
  """Seperates the sentence into tokens.

  Args:
    sentence: a python string denoting the sentence.

  Returns:
    tokens: a list of strings denoting extracted tokens.
  """
  sentence = sentence.replace('<UNK>', 'UNKDENSECAP')
  tokens = nltk.word_tokenize(sentence.lower())
  return tokens


def _load_image_path_list(image_data_path):
  """Loads image paths from the image_data_path.

  Args:
    image_data_path: path to the directory saving images.

  Returns:
    examples: a list of (image_id, filename) tuples.
  """
  examples = []
  for dirpath, dirnames, filenames in os.walk(image_data_path):
    for filename in filenames:
      image_id = int(filename.split('.')[0])
      filename = os.path.join(dirpath, filename)
      examples.append((image_id, filename))
  return examples


def _load_npy(filename):
  """Loads data in npy format.

  Args:
    filename: Path to the json file.

  Returns:
    a numpy array representing the multi-dimensional data.
  """
  with open(filename, 'rb') as fid:
    return np.load(fid)


def _load_json(filename):
  """Loads data in json format.

  Args:
    filename: Path to the json file.

  Returns:
    a python dict representing the json object.
  """
  with open(filename, 'r') as fid:
    return json.load(fid)


def _decode_qa(data):
  """Decodes groundtruth annotations and candidate questions.

  Args:
    data: a python dict containing keys of `groundtruth_list` and `question_list`.

  Returns:
    gt_strings: a list of tokens representing the words appears in all the groundtruth statements.
    gt_offset: a list of integers representing the offsets of the annotations.
    gt_length: a list of integers representing the lengths of the annotations.
    ca_strings: a list of tokens representing the words appears in all the candidates.
    ca_offset: a list of integers representing the offsets of the candidates.
    ca_length: a list of integers representing the lengths of the candidates.
  """

  def _decode_text(sentences):
    text_string = []
    text_offset = []
    text_length = []
    for sentence in sentences:
      tokens = _tokenize(sentence)
      text_offset.append(len(text_string))
      text_length.append(len(tokens))
      text_string.extend(tokens)
    return text_string, text_offset, text_length

  gt_string = []
  gt_offset = []
  gt_length = []

  if len(data['groundtruth_list']) > 0:
    gt_string, gt_offset, gt_length = _decode_text(data['groundtruth_list'])

  ca_string, ca_offset, ca_length = _decode_text(data['question_list'])

  return {
      'gt_string': gt_string,
      'gt_offset': gt_offset,
      'gt_length': gt_length,
      'ca_string': ca_string,
      'ca_offset': ca_offset,
      'ca_length': ca_length,
  }


def _decode_box_and_text(data):
  """Decodes bounding boxes and texts associated with them.

  Note: text_string[text_offset[i]: text_offset[i] + text_length[i]] denotes the text for the `i`th box.

  Args:
    data: a python dict containing keys of `text` and `paragraphs`.

  Returns:
    number_of_boxes: an integer denoting the number of boxes.
    number_of_tokens: an integer denoting the number of tokens.
    ymin: a list of float representing ymin coordinate.
    xmin: a list of float representing xmin coordinate.
    ymax: a list of float representing ymax coordinate.
    xmax: a list of float representing xmax coordinate.
    text_string: a list of tokens representing the words appears in all bounding boxes.
    text_offset: a list of integers representing the offsets of the texts.
    text_length: a list of integers representing the lengths of the texts.
  """
  ymin = []
  xmin = []
  ymax = []
  xmax = []
  text_string = []
  text_offset = []
  text_length = []
  for paragraph in data['paragraphs']:
    ymin.append(paragraph['bounding_box']['ymin'])
    xmin.append(paragraph['bounding_box']['xmin'])
    ymax.append(paragraph['bounding_box']['ymax'])
    xmax.append(paragraph['bounding_box']['xmax'])
    tokens = _tokenize(paragraph['text'])
    text_offset.append(len(text_string))
    text_length.append(len(tokens))
    text_string.extend(tokens)

  return {
      'number_of_boxes': len(data['paragraphs']),
      'number_of_tokens': len(text_string),
      'ymin': ymin,
      'xmin': xmin,
      'ymax': ymax,
      'xmax': xmax,
      'text_string': text_string,
      'text_offset': text_offset,
      'text_length': text_length,
  }


def _add_box_and_text(tf_example, scope, data):
  """Adds box-level caption annotations to the tf.train.Example proto.

  Args:
    tf_example: an instance of tf.train.Example.
    scope: name in the tf.train.Example.
    data: 
  """
  feature_map = tf_example.features.feature

  for name in ['number_of_boxes', 'number_of_tokens']:
    feature_map[scope + '/' + name].int64_list.CopyFrom(
        tf.train.Int64List(value=[data[name]]))

  for name in ['ymin', 'xmin', 'ymax', 'xmax']:
    feature_map[scope + '/bbox/' + name].float_list.CopyFrom(
        tf.train.FloatList(value=data[name]))

  for name in ['offset', 'length']:
    feature_map[scope + '/text/' + name].int64_list.CopyFrom(
        tf.train.Int64List(value=data['text_' + name]))

  for name in ['string']:
    feature_map[scope + '/text/' + name].bytes_list.CopyFrom(
        tf.train.BytesList(
            value=[v.encode('utf8') for v in data['text_' + name]]))
  return tf_example


def _dict_to_tf_example(data):
  """Converts python dict to tf example.

  Args:
    data: the python dict returned by `_load_annotation`.

  Returns:
    tf_example: the tf.train.Example proto.
  """
  # Add the basic image feature.
  tf_example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image_id':
              tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[data['image_id']])),
              'feature/img/value':
              tf.train.Feature(
                  float_list=tf.train.FloatList(value=data['img_f'].tolist())),
              'feature/roi/value':
              tf.train.Feature(
                  float_list=tf.train.FloatList(
                      value=data['roi_f'].flatten().tolist())),
              'feature/roi/length':
              tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[data['roi_f'].shape[0]])),
              'feature/symbol/value':
              tf.train.Feature(
                  float_list=tf.train.FloatList(value=data['symbol'].tolist())),
          }))
  feature_map = tf_example.features.feature

  # Add the OCR and the DENSECAP annotations.
  ocr_data = _decode_box_and_text(data['ocr'])
  densecap_data = _decode_box_and_text(data['densecap'])

  tf_example = _add_box_and_text(tf_example, scope='ocr', data=ocr_data)
  tf_example = _add_box_and_text(
      tf_example, scope='densecap', data=densecap_data)

  global _max_ocr_text_len
  global _max_densecap_text_len
  _max_ocr_text_len = max([_max_ocr_text_len] + ocr_data['text_length'])
  _max_densecap_text_len = max([_max_densecap_text_len] +
                               densecap_data['text_length'])

  # Add the QA annotations.
  qa_data = _decode_qa(data['qa'])

  for name in ['offset', 'length']:
    feature_map['label/text/' + name].int64_list.CopyFrom(
        tf.train.Int64List(value=qa_data['gt_' + name]))
    feature_map['question/text/' + name].int64_list.CopyFrom(
        tf.train.Int64List(value=qa_data['ca_' + name]))

  for name in ['string']:
    feature_map['label/text/' + name].bytes_list.CopyFrom(
        tf.train.BytesList(
            value=[v.encode('utf8') for v in qa_data['gt_' + name]]))
    feature_map['question/text/' + name].bytes_list.CopyFrom(
        tf.train.BytesList(
            value=[v.encode('utf8') for v in qa_data['ca_' + name]]))

  return tf_example


def _load_annotation(image_id, unused_filename):
  """Loads the annotation for the image.

  Args:
    image_id: the numeric id of the image.
    unused_filename: path to the image file.

  Returns:
    a python dict containing the annotations.
  """
  json_filename = '{}.json'.format(image_id)
  npy_filename = '{}.npy'.format(image_id)

  data = {}
  data['image_id'] = image_id
  data['img_f'] = _load_npy(
      os.path.join(FLAGS.img_feature_npy_path, npy_filename))
  data['roi_f'] = _load_npy(
      os.path.join(FLAGS.roi_feature_npy_path, npy_filename))
  data['qa'] = _load_json(os.path.join(FLAGS.qa_json_path, json_filename))
  data['ocr'] = _load_json(os.path.join(FLAGS.ocr_json_path, json_filename))
  data['densecap'] = _load_json(
      os.path.join(FLAGS.densecap_json_path, json_filename))
  data['symbol'] = _load_npy(
      os.path.join(FLAGS.symbol_npy_path, npy_filename))
  return data


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  examples = _load_image_path_list(FLAGS.image_data_path)
  tf.logging.info('Load %s examples.', len(examples))

  writers = []
  for i in range(FLAGS.number_of_parts):
    filename = FLAGS.tfrecord_output_path + '-%05d-of-%05d' % (
        i, FLAGS.number_of_parts)
    writers.append(tf.python_io.TFRecordWriter(filename))

  for index, (image_id, filename) in enumerate(examples):
    data = _load_annotation(image_id, filename)
    tf_example = _dict_to_tf_example(data)
    writers[index % FLAGS.number_of_parts].write(tf_example.SerializeToString())

    if index % 100 == 0:
      tf.logging.info('On image %i/%i', index, len(examples))

  for writer in writers:
    writer.close()

  tf.logging.info('_max_ocr_text_len=%i', _max_ocr_text_len)
  tf.logging.info('_max_densecap_text_len=%i', _max_densecap_text_len)


if __name__ == '__main__':
  tf.app.run()
