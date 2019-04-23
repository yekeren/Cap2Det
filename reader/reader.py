from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from protos import reader_pb2
from reader import wsod_reader
from reader import advise_reader


def get_input_fn(options):
  """Returns a function that generate input examples.

  Args:
    options: an instance of reader_pb2.Reader.

  Returns:
    input_fn: a callable that returns a dataset.
  """
  if not isinstance(options, reader_pb2.Reader):
    raise ValueError('options has to be an instance of Reader.')

  reader_oneof = options.WhichOneof('reader_oneof')

  if 'wsod_reader' == reader_oneof:
    return wsod_reader.get_input_fn(options.wsod_reader)

  if 'advise_reader' == reader_oneof:
    return advise_reader.get_input_fn(options.advise_reader)

  raise ValueError('Invalid reader %s' % (reader_oneof))
