#!/bin/sh

set -x

mkdir -p log

# Export PYTHONPATH to use tensorflow models.

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

name="cap2det_coco17_extend_match"

PIPELINE_CONFIG_PATH="configs.iccv.coco/${name}.pbtxt"
MODEL_DIR="ICCV-COCO-logs/${name}"
VOCAB_FILE="configs/coco_vocab.txt"
EVALUATOR="coco"
DETECTION_RESULT_DIR="coco.results/${name}"
SHARDS=4

mkdir -p "${MODEL_DIR}/eval_det2"
mkdir -p "${DETECTION_RESULT_DIR}"

#i=0
#export CUDA_VISIBLE_DEVICES=0
#python "train/predict.py" \
#  --alsologtostderr \
#  --run_once \
#  --input_pattern="output/coco17_testdev.record*" \
#  --shard_indicator="${i}/${SHARDS}" \
#  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
#  --model_dir="${MODEL_DIR}" \
#  --eval_log_dir="tmp/eval_det2" \
#  --saved_ckpts_dir="${MODEL_DIR}/saved_ckpts" \
#  --max_eval_examples=500000 \
#  --evaluator="${EVALUATOR}" \
#  --detection_result_dir="${DETECTION_RESULT_DIR}" \
#  --vocabulary_file="${VOCAB_FILE}" \
#  >> "log/cocoe${i}.log" 2>&1 &
#
#i=1
#export CUDA_VISIBLE_DEVICES=1
#python "train/predict.py" \
#  --alsologtostderr \
#  --run_once \
#  --input_pattern="output/coco17_testdev.record*" \
#  --shard_indicator="${i}/${SHARDS}" \
#  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
#  --model_dir="${MODEL_DIR}" \
#  --eval_log_dir="tmp/eval_det2" \
#  --saved_ckpts_dir="${MODEL_DIR}/saved_ckpts" \
#  --max_eval_examples=500000 \
#  --evaluator="${EVALUATOR}" \
#  --detection_result_dir="${DETECTION_RESULT_DIR}" \
#  --vocabulary_file="${VOCAB_FILE}" \
#  >> "log/cocoe${i}.log" 2>&1 &
#
#i=2
#export CUDA_VISIBLE_DEVICES=2
#python "train/predict.py" \
#  --alsologtostderr \
#  --run_once \
#  --input_pattern="output/coco17_testdev.record*" \
#  --shard_indicator="${i}/${SHARDS}" \
#  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
#  --model_dir="${MODEL_DIR}" \
#  --eval_log_dir="tmp/eval_det2" \
#  --saved_ckpts_dir="${MODEL_DIR}/saved_ckpts" \
#  --max_eval_examples=500000 \
#  --evaluator="${EVALUATOR}" \
#  --detection_result_dir="${DETECTION_RESULT_DIR}" \
#  --vocabulary_file="${VOCAB_FILE}" \
#  >> "log/cocoe${i}.log" 2>&1 &

i=3
export CUDA_VISIBLE_DEVICES=4
python "train/predict.py" \
  --alsologtostderr \
  --run_once \
  --input_pattern="output/coco17_testdev.record*" \
  --shard_indicator="${i}/${SHARDS}" \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --model_dir="${MODEL_DIR}" \
  --eval_log_dir="tmp/eval_det2" \
  --saved_ckpts_dir="${MODEL_DIR}/saved_ckpts" \
  --max_eval_examples=500000 \
  --evaluator="${EVALUATOR}" \
  --detection_result_dir="${DETECTION_RESULT_DIR}" \
  --vocabulary_file="${VOCAB_FILE}" \
  >> "log/cocoe${i}.log" 2>&1 &

exit 0
