#!/bin/sh

set -x

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"
export PYTHONPATH="`pwd`:$PYTHONPATH"

mkdir -p log

name="rpn_only_v3"

PIPELINE_CONFIG_PATH="rpn_configs/${name}.pbtxt"
MODEL_DIR="rpn_logs/${name}"

export CUDA_VISIBLE_DEVICES=1
python object_detection/legacy/train.py \
  --alsologtostderr \
  --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
  --train_dir="${MODEL_DIR}/train" \
  >> "log/${name}_train.log" 2>&1 &

export CUDA_VISIBLE_DEVICES=2
python object_detection/legacy/proposal_eval.py \
  --alsologtostderr \
  --evaluate_proposals \
  --max_proposals=5 \
  --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
  --checkpoint_dir="${MODEL_DIR}/train" \
  --eval_dir="${MODEL_DIR}/eval" \
>> "log/${name}_eval.log" 2>&1 &

exit 0
