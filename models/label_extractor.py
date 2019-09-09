from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
from core.standard_fields import InputDataFields
from protos import label_extractor_pb2


class LabelExtractor(abc.ABC):
  """Label extractor."""

  def __init__(self, options):
    """Initializes the label extractor."""
    self._options = options
    self._classes = None
    self._num_classes = None

  @property
  def classes(self):
    """Returns the classes."""
    return self._classes

  @property
  def num_classes(self):
    """Returns the number of classes."""
    return self._num_classes

  @abc.abstractmethod
  def extract_labels(self, examples):
    """Extracts the pseudo labels."""
    pass


def _match_labels(class_texts, vocabulary_list):
  """Matches labels from texts.

  Args:
    class_texts: A [batch, num_proposal] string tensor.

  Returns:
    A [batch, num_classes] float tensor.
  """
  keys = [class_name for class_id, class_name in enumerate(vocabulary_list)]
  values = [class_id for class_id, class_name in enumerate(vocabulary_list)]
  table = tf.contrib.lookup.HashTable(
      initializer=tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
      default_value=len(vocabulary_list))  # Class ID for Out-of-Vocabulary words.
  ids = table.lookup(class_texts)
  labels = tf.one_hot(indices=ids, depth=1 + len(vocabulary_list), dtype=tf.float32)
  labels = tf.reduce_max(labels, axis=1)[:, :-1]
  return labels

  # categorical_col = tf.feature_column.categorical_column_with_vocabulary_list(
  #     key='class_texts', vocabulary_list=vocabulary_list, num_oov_buckets=1)
  # indicator_col = tf.feature_column.indicator_column(categorical_col)
  # indicator = tf.feature_column.input_layer(
  #     features={'class_texts': class_texts}, feature_columns=[indicator_col])
  # return tf.cast(indicator[:, :-1] > 0, tf.float32)


class GroundtruthExtractor(LabelExtractor):
  """Groundtruth extractor.

  Extracts labels from the ground-truth object annotations.
  """

  def __init__(self, options):
    """Initializes the label extractor."""
    super(GroundtruthExtractor, self).__init__(options)
    with tf.gfile.GFile(options.label_file, "r") as fid:
      self._classes = [line.strip('\n') for line in fid.readlines()]
    self._num_classes = len(self._classes)

  def extract_labels(self, examples):
    """Extracts the pseudo labels.

    Args:
      examples: A dictionary involving image-level annotations.

    Returns:
      labels: A [batch, num_classes] tensor denoting the presence of classes.
    """
    with tf.name_scope('groundtruth_extractor'):
      return _match_labels(
          class_texts=examples[InputDataFields.object_texts],
          vocabulary_list=self._classes)


class ExactMatchExtractor(LabelExtractor):
  """ExactMatch extractor.

  Extracts labels from captions using ExactMatch.
  Only processes the basic substutions using a synonyms dictionary.
  """

  def __init__(self, options):
    """Initializes the label extractor."""
    super(ExactMatchExtractor, self).__init__(options)
    with tf.gfile.GFile(options.label_file, "r") as fid:
      self._classes = [line.strip('\n') for line in fid.readlines()]
    self._num_classes = len(self._classes)

  def extract_labels(self, examples):
    """Extracts the pseudo labels.
    Args:
      examples: A dictionary involving image-level annotations.
    Returns:
      labels: A [batch, num_classes] tensor denoting the presence of classes.
    """
    synonyms = {
        'traffic light': 'stoplight',
        'fire hydrant': 'hydrant',
        'stop sign': 'sign',
        'parking meter': 'meter',
        'sports ball': 'ball',
        'baseball bat': 'bat',
        'baseball glove': 'glove',
        'tennis racket': 'racket',
        'wine glass': 'wineglass',
        'hot dog': 'hotdog',
        'potted plant': 'plant',
        'dining table': 'table',
        'cell phone': 'cellphone',
        'teddy bear': 'teddy',
        'hair drier': 'hairdryer',
    }
    classes_to_match = [synonyms.get(x, x) for x in self._classes]

    with tf.name_scope('exact_match_extractor'):
      return _match_labels(
          class_texts=examples[InputDataFields.concat_caption_string],
          vocabulary_list=classes_to_match)


class ExtendMatchExtractor(LabelExtractor):
  """ExtendMatch extractor.

  Extracts labels from captions using ExtendMatch.
  Treats synonyms as matches.
  """

  def __init__(self, options):
    """Initializes the label extractor."""
    super(ExtendMatchExtractor, self).__init__(options)

    self._name2id = {}
    self._classes = []

    with tf.gfile.GFile(options.label_file, "r") as fid:
      for class_id, line in enumerate(fid):
        class_name, synonyms = line.strip('\n').split('\t')
        self._name2id[class_name] = class_id
        self._classes.append(class_name)
        synonyms = [x for x in synonyms.split(',') if x]
        if synonyms:
          for synonym in synonyms:
            self._name2id[synonym] = class_id
        else:
          tf.logging.warning('Class %s has no synonym.', class_name)
    self._num_classes = len(self._classes)

  def extract_labels(self, examples):
    """Extracts the pseudo labels.

    Args:
      examples: A dictionary involving image-level annotations.

    Returns:
      labels: A [batch, num_classes] tensor denoting the presence of classes.
    """
    with tf.name_scope('extend_match_extractor'):
      items = self._name2id.items()
      keys = [k for k, v in items]
      values = [v for k, v in items]
      table = tf.contrib.lookup.HashTable(
          initializer=tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
          default_value=self.num_classes)  # Class ID for Out-of-Vocabulary words.
      ids = table.lookup(examples[InputDataFields.concat_caption_string])
      labels = tf.one_hot(indices=ids, depth=1 + self.num_classes, dtype=tf.float32)
      labels = tf.reduce_max(labels, axis=1)[:, :-1]
    return labels


def build_label_extractor(config):
  """Builds label extractor according to the config.
  Args:
    config: An instance of label_extractor_pb2.LabelExtractor.
  Returns:
    An instance of LabelExtractor.
  """
  if not isinstance(config, label_extractor_pb2.LabelExtractor):
    raise ValueError('Config has to be an instance of LabelExtractor proto.')

  label_extractor_oneof = config.WhichOneof('label_extractor_oneof')

  if 'groundtruth_extractor' == label_extractor_oneof:
    return GroundtruthExtractor(config.groundtruth_extractor)

  elif 'exact_match_extractor' == label_extractor_oneof:
    return ExactMatchExtractor(config.exact_match_extractor)

  elif 'extend_match_extractor' == label_extractor_oneof:
    return ExtendMatchExtractor(config.extend_match_extractor)

  elif 'word_vector_match_extractor' == label_extractor_oneof:
    return WordVectorMatchExtractor(config.word_vector_match_extractor)

  raise ValueError('Invalid label extractor %s' % label_extractor_oneof)
