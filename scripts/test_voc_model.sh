#!/bin/sh

set -x

mkdir -p log

# Export PYTHONPATH to use tensorflow models.

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

# Single host.

HOST1="127.0.0.1"

name="per_class_ssquality_12"

EVALUATOR="pascal"
VOCAB_FILE="configs/voc_vocab.txt"
PIPELINE_CONFIG_PATH="configs.iccv.voc/${name}.pbtxt"
MODEL_DIR="ICCV-VOC-logs/${name}"

mkdir -p "${MODEL_DIR}/eval_det2"

type="evaluator"
index=0

export CUDA_VISIBLE_DEVICES=2
python "train/predict.py" \
  --alsologtostderr \
  --run_once \
  --input_pattern="output/VOC2012_trainval_ssbox_quality.record*" \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --model_dir="${MODEL_DIR}" \
  --eval_log_dir="tmp/eval_det2" \
  --saved_ckpts_dir="tmp/saved_ckpts" \
  --results_dir="results.trainval" \
  --max_eval_examples=12000 \
  --evaluator="${EVALUATOR}" \
  --vocabulary_file="${VOCAB_FILE}" \
  >> "log/${name}.voc.log" 2>&1 &


exit 0
