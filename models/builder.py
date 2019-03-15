from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import model_pb2
from models import gap_model
from protos import gap_model_pb2
from models import cam_model
from protos import cam_model_pb2
from models import mil_model
from protos import mil_model_pb2
from models import oicr_model
from protos import oicr_model_pb2
from models import oicr_dilated_model
from protos import oicr_dilated_model_pb2
from models import multi_resol_model
from protos import multi_resol_model_pb2
from models import frcnn_model
from protos import frcnn_model_pb2
from models import voc_model
from protos import voc_model_pb2
from models import wsod_voc_model
from protos import wsod_voc_model_pb2
from models import nod_model
from protos import nod_model_pb2
from models import nod2_model
from protos import nod2_model_pb2
from models import nod3_model
from protos import nod3_model_pb2
from models import nod4_model
from protos import nod4_model_pb2
from models import nod5_model
from protos import nod5_model_pb2
from models import visual_w2v_model
from protos import visual_w2v_model_pb2
from models import stacked_attn_model
from protos import stacked_attn_model_pb2
from models import text_classification_model
from protos import text_classification_model_pb2


def build(options, is_training=False):
  """Builds a Model based on the options.

  Args:
    options: a model_pb2.Model instance.
    is_training: True if this model is being built for training.

  Returns:
    a Model instance.

  Raises:
    ValueError: if options is invalid.
  """
  if not isinstance(options, model_pb2.Model):
    raise ValueError('The options has to be an instance of model_pb2.Model.')

  if options.HasExtension(gap_model_pb2.GAPModel.ext):
    return gap_model.Model(options.Extensions[gap_model_pb2.GAPModel.ext],
                           is_training)

  if options.HasExtension(cam_model_pb2.CAMModel.ext):
    return cam_model.Model(options.Extensions[cam_model_pb2.CAMModel.ext],
                           is_training)

  if options.HasExtension(mil_model_pb2.MILModel.ext):
    return mil_model.Model(options.Extensions[mil_model_pb2.MILModel.ext],
                           is_training)

  if options.HasExtension(oicr_model_pb2.OICRModel.ext):
    return oicr_model.Model(options.Extensions[oicr_model_pb2.OICRModel.ext],
                            is_training)

  if options.HasExtension(oicr_dilated_model_pb2.OICRDilatedModel.ext):
    return oicr_dilated_model.Model(
        options.Extensions[oicr_dilated_model_pb2.OICRDilatedModel.ext],
        is_training)

  if options.HasExtension(multi_resol_model_pb2.MultiResolModel.ext):
    return multi_resol_model.Model(
        options.Extensions[multi_resol_model_pb2.MultiResolModel.ext],
        is_training)

  if options.HasExtension(frcnn_model_pb2.FRCNNModel.ext):
    return frcnn_model.Model(options.Extensions[frcnn_model_pb2.FRCNNModel.ext],
                             is_training)

  if options.HasExtension(voc_model_pb2.VOCModel.ext):
    return voc_model.Model(options.Extensions[voc_model_pb2.VOCModel.ext],
                           is_training)

  if options.HasExtension(wsod_voc_model_pb2.WsodVocModel.ext):
    return wsod_voc_model.Model(
        options.Extensions[wsod_voc_model_pb2.WsodVocModel.ext], is_training)

  if options.HasExtension(nod_model_pb2.NODModel.ext):
    return nod_model.Model(options.Extensions[nod_model_pb2.NODModel.ext],
                           is_training)

  if options.HasExtension(nod2_model_pb2.NOD2Model.ext):
    return nod2_model.Model(options.Extensions[nod2_model_pb2.NOD2Model.ext],
                            is_training)

  if options.HasExtension(nod3_model_pb2.NOD3Model.ext):
    return nod3_model.Model(options.Extensions[nod3_model_pb2.NOD3Model.ext],
                            is_training)

  if options.HasExtension(nod4_model_pb2.NOD4Model.ext):
    return nod4_model.Model(options.Extensions[nod4_model_pb2.NOD4Model.ext],
                            is_training)

  if options.HasExtension(nod5_model_pb2.NOD5Model.ext):
    return nod5_model.Model(options.Extensions[nod5_model_pb2.NOD5Model.ext],
                            is_training)

  if options.HasExtension(visual_w2v_model_pb2.VisualW2vModel.ext):
    return visual_w2v_model.Model(
        options.Extensions[visual_w2v_model_pb2.VisualW2vModel.ext],
        is_training)

  if options.HasExtension(stacked_attn_model_pb2.StackedAttnModel.ext):
    return stacked_attn_model.Model(
        options.Extensions[stacked_attn_model_pb2.StackedAttnModel.ext],
        is_training)

  if options.HasExtension(text_classification_model_pb2.TextClassificationModel.ext):
    return text_classification_model.Model(
        options.Extensions[text_classification_model_pb2.TextClassificationModel.ext],
        is_training)
  raise ValueError('Unknown model: {}'.format(model))
