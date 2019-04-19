#!/bin/sh

set -x

mkdir -p mingda_log

# Export PYTHONPATH to use tensorflow models.

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

# for ((i=0;i<6;++i)); do
# export CUDA_VISIBLE_DEVICES=$((i%6))
python "tools/gather_faster_rcnn_voc2012_test_results.py" --alsologtostderr 
  # >> "mingda_log/voc12.test.submission.log" 2>&1 &
# done
#   --shard_indicator="${i}/6" \
#   --saved_ckpts_dir="${MODEL_DIR}/saved_ckpts" \

exit 0

