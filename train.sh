#!/bin/sh

set -x

# Export PYTHONPATH to use tensorflow models.

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

NAME="voc07_weighted"
export CUDA_VISIBLE_DEVICES=2

PIPELINE_CONFIG_PATH="configs/${NAME}.pbtxt"
MODEL_DIR="logs/${NAME}"

#python "train/predict.py" \
#  --alsologtostderr \
#  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
#  --model_dir="${MODEL_DIR}" \
#  --eval_log_dir="${MODEL_DIR}/eval_det" \
#  --saved_ckpts_dir="${MODEL_DIR}/saved_ckpts" \
#  --max_eval_examples=500 \
#  --max_visl_examples=100 \
#  --min_eval_steps=0 \
#  --evaluator="pascal" \
#  --visl_file_path="tmp/${NAME}.html" \
#  --vocabulary_file="configs/voc.vocab" \
#  || exit -1

python train/trainer_main.py \
  --alsologtostderr \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --model_dir="${MODEL_DIR}" \
  || exit -1

exit 0
