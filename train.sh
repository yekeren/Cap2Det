#!/bin/sh

set -x

mkdir -p mz_log
mkdir -p mz_tmp

# Export PYTHONPATH to use tensorflow models.

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

NAME="voc07"
export CUDA_VISIBLE_DEVICES=2

PIPELINE_CONFIG_PATH="configs/${NAME}.pbtxt"
MODEL_DIR="logs/${NAME}"

#export CUDA_VISIBLE_DEVICES=1
#python "train/predict2.py" \
#  --alsologtostderr \
#  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
#  --model_dir="${MODEL_DIR}" \
#  --run_once \
#  --eval_log_dir="tmp/" \
#  --saved_ckpts_dir="${MODEL_DIR}/saved_ckpts" \
#  --input_pattern="output/flickr30k_trainval_ssbox_quality.record-?????-of-00020" \
#  --max_eval_examples=500 \
#  --max_visl_examples=500 \
#  --min_eval_steps=0 \
#  --evaluator="coco" \
#  --visl_file_path="tmp/${NAME}.html" \
#  --vocabulary_file="configs/coco_vocab.txt" \
####  >> "log/${NAME}.eval.log" 2>&1 &
###
#exit 0
##

python train/trainer_main.py \
  --alsologtostderr \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --model_dir="${MODEL_DIR}" \
  || exit -1


#python model-tools/compute_average_precision.py \
#  --alsologtostderr \
#  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
#  --model_dir="${MODEL_DIR}"

exit 0
