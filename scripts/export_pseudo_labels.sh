#!/bin/sh

set -x

mkdir -p log

# Export PYTHONPATH to use tensorflow models.

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

NAME="text_classification_mlp2"
PIPELINE_CONFIG_PATH="configs.iccv.coco/${NAME}.pbtxt"
MODEL_DIR="ICCV-TXT-logs/${NAME}"

export CUDA_VISIBLE_DEVICES=2
#python model-tools/export_pseudo_labels.py \
#  --alsologtostderr \
#  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
#  --input_pattern="output/coco17_train.record*" \
#  --output_json_path="output/${NAME}.pseudo.json" \
#  --model_dir="${MODEL_DIR}" \
#  || exit -1
python model-tools/export_pseudo_labels.py \
  --alsologtostderr \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --input_pattern="output/flickr30k_trainval_ssbox_quality.record*" \
  --output_json_path="output/${NAME}.flickr30k.json" \
  --model_dir="${MODEL_DIR}" \
  || exit -1

exit 0
