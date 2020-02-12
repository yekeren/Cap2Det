#!/bin/bash

set -o errexit
set -o nounset
set -x

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 MODEL_NAME"
  exit 1
fi

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

NAME=$1
MODEL_DIR="logs/${NAME}"
PIPELINE_CONFIG_PATH="configs/${NAME}.pbtxt"

if [ ! -f "${PIPELINE_CONFIG_PATH}" ]; then
  echo "Config file ${PIPELINE_CONFIG_PATH} does not exist."
  exit 1
fi


export CUDA_VISIBLE_DEVICES="0"
python "train/trainer_main.py" \
  --alsologtostderr \
  --pipeline_proto=${PIPELINE_CONFIG_PATH} \
  --model_dir="${MODEL_DIR}" \
  > "log/${NAME}.train.log" 2>&1 &

export CUDA_VISIBLE_DEVICES="2"
python "train/predict.py" \
  --alsologtostderr \
  --evaluator="pascal" \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --model_dir="${MODEL_DIR}" \
  --max_eval_examples=500 \
  --max_visl_examples=500 \
  --label_file="data/voc_label.txt" \
  --eval_log_dir="${MODEL_DIR}/eval_det" \
  --saved_ckpts_dir="${MODEL_DIR}/saved_ckpts" \
  --visl_file_path="${MODEL_DIR}/visl.html" \
  >> "log/${NAME}.eval.log" 2>&1 &
exit 0
