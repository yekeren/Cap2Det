from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from google.protobuf import text_format
from core.standard_fields import InputDataFields

from protos import cap2det_model_pb2
from models import cap2det_model


class Cap2DetModelTest(tf.test.TestCase):

  def test_prediction(self):
    pass


if __name__ == '__main__':
  tf.test.main()
