#!/bin/sh

set -x

mkdir -p log

# Export PYTHONPATH to use tensorflow models.

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

name="cap2det_coco17_exact_match_text_clsf"

EVALUATOR="pascal"
VOCAB_FILE="configs/voc_vocab.txt"

PIPELINE_CONFIG_PATH="configs.iccv.coco/${name}.pbtxt"
MODEL_DIR="ICCV-COCO-logs/${name}"

#PIPELINE_CONFIG_PATH="configs.iccv.flickr30k/${name}.pbtxt"
#MODEL_DIR="ICCV-FLICKR30K-logs/${name}"

mkdir -p "${MODEL_DIR}/eval_det2"
#mkdir -p "${DETECTION_RESULT_DIR}"

type="evaluator"
index=0

export CUDA_VISIBLE_DEVICES=3
python "train/predict.py" \
  --alsologtostderr \
  --run_once \
  --eval_best_model \
  --eval_coco_on_voc \
  --input_pattern="output/VOC2007_test_ssbox_quality.record*" \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --model_dir="${MODEL_DIR}" \
  --eval_log_dir="tmp/eval_det2" \
  --saved_ckpts_dir="${MODEL_DIR}/saved_ckpts" \
  --max_eval_examples=5000 \
  --evaluator="${EVALUATOR}" \
  --vocabulary_file="${VOCAB_FILE}" \
  >> "log/${name}.eval_coco_model_on_pascal.log" 2>&1 &

exit 0
