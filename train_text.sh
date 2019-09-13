#!/bin/bash

set -o errexit
set -o nounset
set -x

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 MODEL_NAME"
  exit 1
fi

NAME=$1
MODEL_DIR="logs/${NAME}"
PIPELINE_CONFIG_PATH="configs/${NAME}.pbtxt"

if [ ! -f "${PIPELINE_CONFIG_PATH}" ]; then
  echo "Config file ${PIPELINE_CONFIG_PATH} does not exist."
  exit 1
fi


export CUDA_VISIBLE_DEVICES="3"
python "train/trainer_main.py" \
  --alsologtostderr \
  --pipeline_proto=${PIPELINE_CONFIG_PATH} \
  --model_dir="${MODEL_DIR}" \

exit 0
