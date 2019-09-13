from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from protos import reader_pb2
from readers import cap2det_reader


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

  if 'cap2det_reader' == reader_oneof:
    return cap2det_reader.get_input_fn(options.cap2det_reader)

  raise ValueError('Invalid reader %s' % (reader_oneof))
