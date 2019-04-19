#!/bin/sh

set -x

mkdir -p mingda_log

# Export PYTHONPATH to use tensorflow models.

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

name="per_class_ssquality_07_v6"

PIPELINE_CONFIG_PATH="configs.iccv.voc/${name}.pbtxt"
MODEL_DIR="ICCV-VOC-logs/${name}"
VOCAB_FILE="configs/voc_vocab.txt"
DETECTION_RESULT_DIR="voc2007.trainval.results/${name}"

mkdir -p "${DETECTION_RESULT_DIR}"

# for ((i=0;i<6;++i)); do
# export CUDA_VISIBLE_DEVICES=$((i%6))
export CUDA_VISIBLE_DEVICES="5"
python "train/export.py" \
  --alsologtostderr \
  --input_pattern="output/VOC2007_test_ssbox_quality.record-?????-of-00020" \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --model_dir="${MODEL_DIR}" \
  --detection_results_dir="${DETECTION_RESULT_DIR}" \
  --vocabulary_file="${VOCAB_FILE}" \
  >> "mingda_log/voc07.trainval.${name}.export.log" 2>&1 &
# done
#   --shard_indicator="${i}/6" \
#   --saved_ckpts_dir="${MODEL_DIR}/saved_ckpts" \

exit 0

