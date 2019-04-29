#!/bin/sh

set -x

# Export PYTHONPATH to use tensorflow models.

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

NAME="ads_cap2det_extend"
PIPELINE_CONFIG_PATH="configs/${NAME}.pbtxt"
MODEL_PATH="zoo/${NAME}/model.ckpt-100003"

PART="test"
DETECTION_RESULT_DIR="raw_data/ads_wsod.json/${PART}"
mkdir -p "${DETECTION_RESULT_DIR}"

for ((i=0;i<3;++i)); do
export CUDA_VISIBLE_DEVICES=$i
python "train/export_results.py" \
  --alsologtostderr \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --model_path="${MODEL_PATH}" \
  --input_pattern="output/ads_${PART}_ssbox_quality.record-*" \
  --shard_indicator="${i}/3" \
  --detection_results_dir="${DETECTION_RESULT_DIR}" \
  >> "log/export_${i}.log" 2>&1 &
done

exit 0
