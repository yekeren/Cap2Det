from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class OperationNames(object):
  """Names of operations."""

  # Reader.

  parse_single_example = "parse_single_example"
  decode_image = "decode_image"
  decode_caption = "decode_caption"
  decode_proposal = "decode_proposal"
  decode_bbox = "decode_bbox"

  # Prediction.

  image_model = "image_model"
  text_model = "text_model"
  image_l2_norm = "image_l2_norm"
  text_l2_norm = "text_l2_norm"

  calc_pairwise_similarity = "calc_pairwise_similarity"

  # Training.

  mine_in_batch_triplet = "mine_in_batch_triplet"

  # Validation.

  caption_retrieval = "caption_retrieval"


class TFExampleDataFields(object):
  """Names for the fields defined in the tf::Example."""

  # Basic image information.

  image_id = "image/source_id"
  image_encoded = "image/encoded"

  # Caption annotations.

  caption_string = "image/caption/string"
  caption_offset = "image/caption/offset"
  caption_length = "image/caption/length"

  # Bounding box annotations.
  number_of_proposals = "image/proposal/num_proposals"
  proposal_box = "image/proposal/bbox"
  proposal_box_ymin = "image/proposal/bbox/ymin"
  proposal_box_xmin = "image/proposal/bbox/xmin"
  proposal_box_ymax = "image/proposal/bbox/ymax"
  proposal_box_xmax = "image/proposal/bbox/xmax"

  object_box = "image/object/bbox"
  object_text = "image/object/class/text"
  object_label = "image/object/class/label"

  object_box_ymin = "image/object/bbox/ymin"
  object_box_xmin = "image/object/bbox/xmin"
  object_box_ymax = "image/object/bbox/ymax"
  object_box_xmax = "image/object/bbox/xmax"


class InputDataFields(object):
  """Names of the input tensors."""
  image = "image"
  image_id = "image_id"

  # image_height / image_width denote the original image shape.

  image_height = "image_height"
  image_width = "image_width"

  # image_shape in the batch.

  image_shape = "image_shape"

  num_captions = "num_captions"
  caption_strings = "caption_strings"
  caption_lengths = "caption_lengths"
  category_strings = "caption_strings"

  concat_caption_string = "concat_caption_string"
  concat_caption_length = "concat_caption_length"

  num_objects = 'number_of_objects'
  object_boxes = 'object_boxes'
  object_texts = 'object_texts'

  proposals = 'proposals'
  num_proposals = 'number_of_proposals'


class DetectionResultFields(object):
  """Names of the output detection tensors."""
  num_proposals = 'num_proposals'
  proposal_boxes = 'proposal_boxes'
  proposal_scores = 'proposal_scores'

  class_labels = 'class_labels'

  num_detections = 'num_detections'
  detection_boxes = 'detection_boxes'
  detection_scores = 'detection_scores'
  detection_classes = 'detection_classes'


class GAPPredictionTasks(object):
  """Prediction tasks of the GAP model."""
  similarity = "similarity"
  image_saliency = "image_saliency"
  image_score_map = "image_score_map"
  word_saliency = "word_saliency"


class GAPVariableScopes(object):
  """Variable scopes used in GAP model."""
  cnn = "CNN"
  image_proj = "image_proj"
  word_embedding = "coco_word_embedding"
  image_saliency = "image_saliency"
  word_saliency = "word_saliency"


class GAPPredictions(object):
  """Predictions in the GAP model."""
  image_id = "image_id"
  image_ids_gathered = "image_ids_gathered"
  similarity = "similarity"

  image_saliency = "image_saliency"
  image_score_map = "image_score_map"
  word_embedding = "word_embedding"
  word_saliency = "image_saliency"
  vocabulary = "vocabulary"


class VOCPredictionTasks(object):
  """Prediction tasks of the VOC model."""
  class_labels = "class_labels"
  image_saliency = "image_saliency"
  image_score_map = "image_score_map"


class VOCVariableScopes(object):
  """Variable scopes used in VOC model."""
  cnn = "Cnn"
  image_proj = "image_proj"


class VOCPredictions(object):
  """Predictions in the VOC model."""
  class_label = "class_label"
  class_act_map = "class_act_map"


class CAMTasks(object):
  """Prediction tasks of the CAM model."""
  image_label = "image_label"
  class_act_map = "image_score_map"


class CAMVariableScopes(object):
  """Variable scopes used in CAM model."""
  cnn = "Cnn"
  image_proj = "image_proj"
  image_proj_second = "image_proj_second"


class CAMPredictions(object):
  """Predictions in the CAM model."""
  class_act_map_list = "class_act_map_list"
  class_act_map = "image_score_map"
  labels = "labels"
  first_stage_logits_list = "first_stage_logits_list"
  second_stage_logits = "second_stage_logits"
  proposals = "proposals"
  proposal_scores = "proposal_scores"


class StackedAttnPredictions(object):
  logits = 'logits'


class NODPredictions(object):
  """Predictions in the NOD model."""
  midn_class_scores = 'midn_class_scores'
  oicr_proposal_scores = 'oicr_proposal_scores'
  midn_proba_h_given_c = 'midn_proba_h_given_c'


class NOD2Predictions(object):
  """Predictions in the NOD model."""
  midn_class_logits = 'midn_class_logits'
  midn_class_scores_sigmoid = 'midn_class_scores_sigmoid'
  midn_class_scores_softmax = 'midn_class_scores_softmax'
  oicr_proposal_scores = 'oicr_proposal_scores'

  midn_proba_r_given_c = 'midn_proba_r_given_c'
  midn_proba_h_given_c = 'midn_proba_h_given_c'


class NOD3Predictions(object):
  """Predictions in the NOD model."""
  midn_class_logits = 'midn_class_logits'
  midn_class_scores_sigmoid = 'midn_class_scores_sigmoid'
  midn_class_scores_softmax = 'midn_class_scores_softmax'
  oicr_proposal_scores = 'oicr_proposal_scores'

  midn_proba_r_given_c = 'midn_proba_r_given_c'
  midn_proba_h_given_c = 'midn_proba_h_given_c'

  training_only_caption_strings = 'train_only_caption_strings'
  training_only_caption_lengths = 'train_only_caption_lengths'

  image_id = 'image_id'
  similarity = 'similarity'


class NOD4Predictions(object):
  """Predictions in the NOD model."""
  midn_class_logits = 'midn_class_logits'
  midn_class_scores_sigmoid = 'midn_class_scores_sigmoid'
  midn_class_scores_softmax = 'midn_class_scores_softmax'
  oicr_proposal_scores = 'oicr_proposal_scores'

  midn_proba_r_given_c = 'midn_proba_r_given_c'
  midn_proba_h_given_c = 'midn_proba_h_given_c'

  #training_only_caption_strings = 'train_only_caption_strings'
  #training_only_caption_lengths = 'train_only_caption_lengths'

  debug_groundtruth_labels = 'debug_groundtruth_labels'
  debug_pseudo_labels = 'debug_pseudo_labels'

  image_id = 'image_id'
  similarity = 'similarity'


class NOD5Predictions(object):
  """Predictions in the NOD model."""
  midn_class_logits = 'midn_class_logits'
  midn_class_scores_sigmoid = 'midn_class_scores_sigmoid'
  midn_class_scores_softmax = 'midn_class_scores_softmax'
  oicr_proposal_scores = 'oicr_proposal_scores'

  midn_proba_r_given_c = 'midn_proba_r_given_c'
  midn_proba_h_given_c = 'midn_proba_h_given_c'

  training_only_caption_strings = 'train_only_caption_strings'
  training_only_caption_lengths = 'train_only_caption_lengths'

  image_id = 'image_id'
  similarity = 'similarity'

  predicted_logits = 'predicted_logits'


class VisualW2vPredictions(object):
  """Predictions in the NOD model."""
  midn_class_logits = 'midn_class_logits'
  midn_class_scores_sigmoid = 'midn_class_scores_sigmoid'
  midn_class_scores_softmax = 'midn_class_scores_softmax'
  oicr_proposal_scores = 'oicr_proposal_scores'

  midn_proba_r_given_c = 'midn_proba_r_given_c'
  midn_proba_h_given_c = 'midn_proba_h_given_c'

  image_id = 'image_id'
  image_ids_gathered = 'image_ids_gathered'
  similarity = 'similarity'
  word2vec = 'word2vec'


class TextClassificationPredictions(object):
  """predictions in the nod model."""
  logits = 'logits'
  midn_class_logits = 'midn_class_logits'
  midn_class_scores_sigmoid = 'midn_class_scores_sigmoid'
  midn_class_scores_softmax = 'midn_class_scores_softmax'
  oicr_proposal_scores = 'oicr_proposal_scores'

  midn_proba_r_given_c = 'midn_proba_r_given_c'
  midn_proba_h_given_c = 'midn_proba_h_given_c'

  image_id = 'image_id'
  image_ids_gathered = 'image_ids_gathered'
  similarity = 'similarity'
  word2vec = 'word2vec'

  vocab = 'vocab'


class OICRTasks(object):
  """Prediction tasks of the OICR model."""
  image_label = "image_label"
  class_act_map = "image_score_map"


class OICRVariableScopes(object):
  """Variable scopes used in OICR model."""
  pass
  #cnn = "cnn"
  #image_proj = "image_proj"
  #image_proj_second = "image_proj_second"


class OICRPredictions(object):
  """Predictions in the OICR model."""
  mipn_logits = 'mipn_logits'
  midn_logits = 'midn_logits'
  midn_proba_r_given_c = 'midn_proba_r_given_c'
  midn_proba_r_given_h = 'midn_proba_r_given_h'
  oicr_proposal_scores = 'oicr_proposal_scores'
  midn_proposal_scores = 'midn_proposal_scores'
  #class_act_map_list = "class_act_map_list"
  #class_act_map = "image_score_map"
  #labels = "labels"
  #first_stage_logits_list = "first_stage_logits_list"
  #second_stage_logits = "second_stage_logits"
  #proposals = "proposals"
  #proposal_scores = "proposal_scores"
