from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import tensorflow as tf
from core.standard_fields import InputDataFields
from core import utils
from protos import label_extractor_pb2

slim = tf.contrib.slim


def _match_labels(class_texts, vocabulary_list):
  """Matches labels from texts.

  Args:
    class_texts: A [batch, num_tokens] string tensor.

  Returns:
    A [batch, num_classes] float tensor.
  """
  keys = [class_name for class_id, class_name in enumerate(vocabulary_list)]
  values = [class_id for class_id, class_name in enumerate(vocabulary_list)]
  table = tf.contrib.lookup.HashTable(
      initializer=tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
      default_value=len(
          vocabulary_list))  # Class ID for Out-of-Vocabulary words.
  ids = table.lookup(class_texts)
  labels = tf.one_hot(
      indices=ids, depth=1 + len(vocabulary_list), dtype=tf.float32)

  batch, num_tokens = utils.get_tensor_shape(class_texts)
  labels = tf.cond(
      num_tokens > 0,
      true_fn=lambda: tf.reduce_max(labels, axis=1)[:, :-1],
      false_fn=lambda: tf.zeros(shape=[batch, len(vocabulary_list)]))
  return labels


def _replace_class_names(class_names):
  """Replaces class names based on a synonym dictionary.

  Args:
    class_names: Original class names.

  Returns:
    A list of new class names.
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
  return [synonyms.get(x, x) for x in class_names]


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
    classes_to_match = _replace_class_names(self._classes)

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
          default_value=self.
          num_classes)  # Class ID for Out-of-Vocabulary words.
      ids = table.lookup(examples[InputDataFields.concat_caption_string])
      labels = tf.one_hot(
          indices=ids, depth=1 + self.num_classes, dtype=tf.float32)

      batch, num_tokens = utils.get_tensor_shape(
          examples[InputDataFields.concat_caption_string])
      labels = tf.cond(
          num_tokens > 0,
          true_fn=lambda: tf.reduce_max(labels, axis=1)[:, :-1],
          false_fn=lambda: tf.zeros(shape=[batch, self.num_classes]))
      return labels


class WordVectorMatchExtractor(LabelExtractor):
  """WordVectorMatch extractor.

  Extracts labels from captions using WordVectorMatch.
  If ExactMatch failed, match the top-1 vector-space synonym.
  """

  def __init__(self, options):
    """Initializes the label extractor."""
    super(WordVectorMatchExtractor, self).__init__(options)
    with tf.gfile.GFile(options.label_file, "r") as fid:
      self._classes = [line.strip('\n') for line in fid.readlines()]
    self._num_classes = len(self._classes)

    # Read the open-vocabulary and its word embedding.
    with tf.gfile.GFile(options.open_vocabulary_file, "r") as fid:
      self._open_vocabulary_list = [
          line.strip('\n') for line in fid.readlines()
      ]
    with open(options.open_vocabulary_word_embedding_file, 'rb') as fid:
      self._open_vocabulary_word_embedding = np.load(fid)

  def _cosine_similarity(self, class_embs, token_embs):
    """Computes the cosine similarity between class and token embeddings.

    Args:
      class_embs: A [num_classes, embedding_dims] tensor.
      token_embs: A [batch, num_tokens, embedding_dims] tensor.

    Returns:
      similarity: A [batch, num_tokens, num_classes] tensor.
    """
    with tf.name_scope('cosine_similarity'):
      class_embs = tf.nn.l2_normalize(class_embs, axis=-1)
      token_embs = tf.nn.l2_normalize(token_embs, axis=-1)

      dot_product = tf.multiply(
          tf.expand_dims(tf.expand_dims(class_embs, axis=0), axis=0),
          tf.expand_dims(token_embs, axis=2))
      return tf.reduce_sum(dot_product, axis=-1)

  def extract_labels(self, examples):
    """Extracts the pseudo labels.
    Args:
      examples: A dictionary involving image-level annotations.
    Returns:
      labels: A [batch, num_classes] tensor denoting the presence of classes.
    """
    init_width = 0.03
    embedding_dims = self._open_vocabulary_word_embedding.shape[-1]

    classes_to_match = _replace_class_names(self._classes)

    # Check if all classes appear in the open-vocabulary.
    for class_name in classes_to_match:
      if not class_name in self._open_vocabulary_list:
        raise ValueError('Class %s has no vector representation.' % class_name)

    with tf.name_scope('word_vector_match_extractor'):

      # Create hash table and word embedding weights.

      table = tf.contrib.lookup.index_table_from_tensor(
          self._open_vocabulary_list, num_oov_buckets=1)
      oov_emb = init_width * (np.random.rand(1, embedding_dims) * 2 - 1)
      embedding_array_data = np.concatenate(
          [self._open_vocabulary_word_embedding, oov_emb], axis=0)
      embedding_weights = tf.get_variable(
          name='weights',
          initializer=embedding_array_data.astype(np.float32),
          trainable=False)  # Freeze the word embedding.

      # Lookup to get the class/token embeddings.

      class_embs = tf.nn.embedding_lookup(
          embedding_weights,
          table.lookup(tf.constant(classes_to_match)),
          max_norm=None)
      token_ids = table.lookup(examples[InputDataFields.concat_caption_string])
      token_embs = tf.nn.embedding_lookup(
          embedding_weights, token_ids, max_norm=None)

      # Compute token-to-class similarity and apply max-pooling.
      # Max-pooling: i.e., treat the top-1 as a match.
      #   similarity shape = [batch, max_num_tokens, num_classes].
      #   similarity_pooled shape = [batch, num_classes]

      batch, num_tokens = utils.get_tensor_shape(
          examples[InputDataFields.concat_caption_string])

      similarity = self._cosine_similarity(class_embs, token_embs)

      oov = len(self._open_vocabulary_list)
      mask = tf.not_equal(token_ids, oov)
      similarity_pooled = utils.masked_maximum(
          data=similarity,
          mask=tf.expand_dims(tf.to_float(mask), axis=-1),
          dim=1)
      similarity_pooled = tf.squeeze(similarity_pooled, axis=1)

      labels_most_similar = tf.one_hot(
          indices=tf.argmax(similarity_pooled, axis=-1),
          depth=self.num_classes,
          dtype=tf.float32)
      labels_most_similar = tf.where(
          tf.reduce_any(mask, axis=-1),
          x=labels_most_similar,
          y=tf.zeros(shape=[batch, self.num_classes]))

      # Consider the exact match.

      labels_exact_match = _match_labels(
          class_texts=examples[InputDataFields.concat_caption_string],
          vocabulary_list=classes_to_match)

      return tf.where(
          tf.reduce_any(labels_exact_match > 0, axis=-1),
          x=labels_exact_match,
          y=labels_most_similar)


class TextClassifierMatchExtractor(LabelExtractor):
  """TextClassifierMatch extractor.

  Extracts labels from captions using TextClassifierMatch.
  If ExactMatch failed, match the top-1 predicted class.
  """

  def __init__(self, options):
    """Initializes the label extractor."""
    super(TextClassifierMatchExtractor, self).__init__(options)
    with tf.gfile.GFile(options.label_file, "r") as fid:
      self._classes = [line.strip('\n') for line in fid.readlines()]
    self._num_classes = len(self._classes)

    # Read the open-vocabulary and its word embedding.
    with tf.gfile.GFile(options.open_vocabulary_file, "r") as fid:
      self._open_vocabulary_list = [
          line.strip('\n') for line in fid.readlines()
      ]
    with open(options.open_vocabulary_word_embedding_file, 'rb') as fid:
      self._open_vocabulary_word_embedding = np.load(fid)

  def _predict(self,
               text_strings,
               text_lengths,
               vocabulary_list,
               vocabulary_word_embedding,
               hidden_units,
               output_units,
               dropout_keep_proba=1.0,
               regularizer=1e-5,
               is_training=False):
    """Predicts labels using the texts.

    Args:
      text_strings: A [batch, num_tokens] string tensor.
      text_lengths: A [batch] int tensor.
      vocabulary_list: A list of string of length vocab_size.
      vocabulary_word_embedding: A [vocab_size, embedding_dims] numpy array.
    """
    # Initial embeddings.

    init_width = 0.03
    oov_emb = init_width * (
        np.random.rand(1, vocabulary_word_embedding.shape[-1]) * 2 - 1)
    embedding_array_data = np.concatenate([vocabulary_word_embedding, oov_emb],
                                          axis=0)

    # Word embedding process.

    with tf.name_scope('word_embedding'):
      table = tf.contrib.lookup.index_table_from_tensor(
          vocabulary_list, num_oov_buckets=1)
      embedding_weights = tf.get_variable(
          name='weights',
          initializer=embedding_array_data.astype(np.float32),
          trainable=False)  # Freeze the word embedding.

      token_ids = table.lookup(text_strings)
      token_embs = tf.nn.embedding_lookup(
          embedding_weights, token_ids, max_norm=None)

    # Multiplayer perceptron.

    with tf.variable_scope('text_classifier'):

      oov = len(vocabulary_list)
      masks = tf.to_float(tf.logical_not(tf.equal(token_ids, oov)))

      hiddens = slim.fully_connected(
          token_embs,
          num_outputs=hidden_units,
          activation_fn=None,
          trainable=is_training,
          weights_regularizer=tf.contrib.layers.l2_regularizer(regularizer),
          scope='layer1')
      hiddens = utils.masked_maximum(
          data=hiddens, mask=tf.expand_dims(masks, axis=-1), dim=1)
      hiddens = tf.squeeze(hiddens, axis=1)
      hiddens = tf.nn.relu(hiddens)
      hiddens = slim.dropout(
          hiddens, dropout_keep_proba, is_training=is_training)

      logits = slim.fully_connected(
          hiddens,
          num_outputs=output_units,
          activation_fn=None,
          trainable=is_training,
          weights_regularizer=tf.contrib.layers.l2_regularizer(regularizer),
          scope='layer2')
    return logits

  def predict(self, examples, is_training=False):
    """Predicts soft labels.
    Args:
      examples: A dictionary involving image-level annotations.
      is_training: A boolean variable.
    Returns:
      labels: A [batch, num_classes] tensor denoting the presence of classes.
    """
    options = self._options
    return self._predict(
        text_strings=examples[InputDataFields.concat_caption_string],
        text_lengths=None,
        vocabulary_list=self._open_vocabulary_list,
        vocabulary_word_embedding=self._open_vocabulary_word_embedding,
        hidden_units=options.hidden_units,
        output_units=self.num_classes,
        dropout_keep_proba=options.dropout_keep_proba,
        is_training=is_training)

  def extract_labels(self, examples):
    """Extracts the pseudo labels.
    Args:
      examples: A dictionary involving image-level annotations.
    Returns:
      labels: A [batch, num_classes] tensor denoting the presence of classes.
    """
    options = self._options

    # Text classifier.

    logits = self.predict(examples, is_training=False)

    tf.train.init_from_checkpoint(
        options.text_classifier_checkpoint_file,
        assignment_map={"text_classifier/": "text_classifier/"})
    logits = tf.stop_gradient(logits)
    probas = tf.sigmoid(logits)

    labels_most_likely = tf.to_float(probas > options.label_threshold)

    # Consider the exact match.

    labels_exact_match = _match_labels(
        class_texts=examples[InputDataFields.concat_caption_string],
        vocabulary_list=self._classes)

    return tf.where(
        tf.reduce_any(labels_exact_match > 0, axis=-1),
        x=labels_exact_match,
        y=labels_most_likely)


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

  elif 'text_classifier_match_extractor' == label_extractor_oneof:
    return TextClassifierMatchExtractor(config.text_classifier_match_extractor)

  raise ValueError('Invalid label extractor %s' % label_extractor_oneof)
