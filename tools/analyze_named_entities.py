from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import nltk
import collections
import numpy as np
import tensorflow as tf

from SPARQLWrapper import SPARQLWrapper, JSON

flags = tf.app.flags

flags.DEFINE_string('query', '', '')

flags.DEFINE_string('slogan_dir', '', '')

flags.DEFINE_string('tfidf_path', '', '')

flags.DEFINE_integer('min_freq', 5, '')

flags.DEFINE_integer('top_k', 10000, '')

flags.DEFINE_string('task', '', '')

flags.DEFINE_string('output_json_path', '', '')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

_stopwords = set(nltk.corpus.stopwords.words('english'))


def _corpus(slogan_dir):
  """Reads the corpus from the path storing slogans.

  Args:
    slogan_dir: Path to the saved slogans.

  Returns:
    A python list, each element is a document.
  """
  count = 0
  corpus = []
  for filename in os.listdir(slogan_dir):
    image_id = int(filename.split('.')[0])
    with open(os.path.join(slogan_dir, filename), 'r') as fid:
      data = json.load(fid)

    count += 1
    if count % 100 == 0:
      tf.logging.info('On %i: %s', count, image_id)

    document = data['text'].replace('\n', ' ')
    corpus.append(document)
  return corpus


def _gather_vocab():
  corpus = _corpus(FLAGS.slogan_dir)
  corpus = [nltk.word_tokenize(document) for document in corpus]
  tf.logging.info("total: %i", len(corpus))

  # Inverse document frequency.

  idf = collections.Counter()
  for document in corpus:
    for token in set(document):
      idf[token] += 1
  idf = dict(((x[0], np.log(1.0 * len(corpus) / (1.0 + x[1])))
              for x in idf.most_common()
              if x[1] >= FLAGS.min_freq))

  # Term frequency.

  tfidf = collections.defaultdict(list)
  for document in corpus:
    counter = collections.Counter()
    for token in document:
      counter[token] += 1
    for x in counter.most_common():
      if 'A' <= x[0][0] <= 'Z' and x[0] in idf and x[0].lower(
      ) not in _stopwords:
        term_freq = 1.0 * x[1]  # / len(document)
        tfidf[x[0]].append(term_freq * idf[x[0]])

  tfidf = [(k, np.max(v)) for k, v in tfidf.items()]
  tfidf = sorted(tfidf, key=lambda x: -x[1])

  with open(FLAGS.tfidf_path, 'w') as fid:
    for k, v in tfidf:
      try:
        k.encode('ascii')
        fid.write('%s\t%.2lf\n' % (k, v))
      except Exception as ex:
        pass

  tf.logging.info('Done')


def _sparql(query):
  query_str = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbpedia2: <http://dbpedia.org/property/>
    PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>

    SELECT DISTINCT ?entry ?comment 
    WHERE {
      {
        ?entry rdfs:label "%s"@en.
        ?entry rdfs:comment ?comment.
        FILTER langMatches(lang(?comment),'en').
      } 
      UNION {
        ?entry dbpedia2:abbreviation ?abbreviation.
        ?entry rdfs:comment ?comment.
        FILTER langMatches(lang(?comment),'en').
        FILTER regex(?abbreviation, "^%s.*?")
      } 
      UNION {
        ?entry_dup rdfs:label "%s"@en.
        ?entry rdfs:comment ?comment.
        ?entry ^dbpedia-owl:wikiPageDisambiguates|^dbpedia-owl:wikiPageRedirects ?entry_dup.
        FILTER langMatches(lang(?comment),'en').
      } 
    }
    LIMIT 20
  """ % (query, query, query)
  sparql = SPARQLWrapper("http://dbpedia.org/sparql")
  sparql.setQuery(query_str)
  sparql.setReturnFormat(JSON)
  results = sparql.query().convert()
  return results


def _sparql_lookup():
  with open(FLAGS.tfidf_path, 'r') as fid:
    word_with_tfidf = [x.strip('\n').split('\t') for x in fid.readlines()]
    queries = [x[0] for x in word_with_tfidf]

  queries = queries[:FLAGS.top_k]
    
  for i, query in enumerate(queries):
    filename = os.path.join(FLAGS.output_json_path, '{}.json'.format(query))

    if not os.path.isfile(filename):
      results = _sparql(query)

      if len(results['results']['bindings']):
        tf.logging.info('Found comment for %s', query)

      try:
        with open(filename, 'w') as fid:
          fid.write(json.dumps(results, indent=2))
      except Exception as ex:
        tf.logging.warn(ex)

      if i % 100 == 0:
        tf.logging.info('On %i/%i', i, len(queries))


def main(_):
  if FLAGS.task == 'gather_vocab':
    _gather_vocab()

  elif FLAGS.task == 'sparql_lookup':
    _sparql_lookup()

  elif FLAGS.task == 'query':
    results = _sparql(FLAGS.query)
    print(json.dumps(results, indent=2))

  else:
    raise ValueError('Invalid task %s' % (FLAGS.task))


if __name__ == '__main__':
  tf.app.run()
