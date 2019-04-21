#!/bin/sh

set -x

# Export PYTHONPATH to use tensorflow models.

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

TMP="./tmp"
NAME="voc07"
export CUDA_VISIBLE_DEVICES=2

PIPELINE_CONFIG_PATH="configs/${NAME}.pbtxt"
MODEL_DIR="logs/${NAME}"

mkdir -p ${TMP}

python "train/predict.py" \
  --alsologtostderr \
  --run_once \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --model_dir="${MODEL_DIR}" \
  --eval_log_dir="${TMP}/${NAME}.eval_det" \
  --saved_ckpts_dir="${TMP}/${NAME}.saved_ckpts" \
  --visl_file_path="${TMP}/${NAME}.html" \
  --results_dir="${TMP}" \
  --max_eval_examples=50000 \
  --evaluator="pascal" \
  --vocabulary_file="configs/voc.vocab" \
  || exit -1

exit 0
