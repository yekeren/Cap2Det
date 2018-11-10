#!/bin/sh

set -x

mkdir -p log

# Export PYTHONPATH to use tensorflow models.

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

name="448_v19"

export CUDA_VISIBLE_DEVICES=2
python train/trainer_main.py \
  --pipeline_proto="configs/${name}.pbtxt" \
  --model_dir="logs/${name}" \
  --logtostderr \
>> log/${name}.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
python object_detection/legacy/proposal_eval.py \
  --alsologtostderr \
  --max_proposals="5" \
  --eval_dir="logs/${name}/proposal_eval/" \
  --score_map_path="logs/${name}/" \
  --pipeline_config_path="configs/${name}_eval.pbtxt" \
>> log/${name}_eval.log 2>&1 &

exit 0

