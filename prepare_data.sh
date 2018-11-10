#!/bin/sh

set -x

TRAIN_IMAGE_FILE="raw_data/train2014.zip"
VAL_IMAGE_FILE="raw_data/val2014.zip"
TEST_IMAGE_FILE="raw_data/test2015.zip"
TRAIN_ANNOTATIONS_FILE="raw_data/annotations/instances_train2014.json"
TRAIN_CAPTION_ANNOTATIONS_FILE="raw_data/annotations/captions_train2014.json"
VAL_ANNOTATIONS_FILE="raw_data/annotations/instances_val2014.json"
VAL_CAPTION_ANNOTATIONS_FILE="raw_data/annotations/captions_val2014.json"
TESTDEV_ANNOTATIONS_FILE="raw_data/annotations/image_info_test-dev2015.json"
OUTPUT_DIR="output"

mkdir -p ${OUTPUT_DIR}

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"

cd "tensorflow_models/research" \
  && protoc object_detection/protos/*.proto --python_out=. \
  && cd - || exit -1

python tools/create_coco_tf_record.py --logtostderr \
  --train_image_file="${TRAIN_IMAGE_FILE}" \
  --val_image_file="${VAL_IMAGE_FILE}" \
  --test_image_file="${TEST_IMAGE_FILE}" \
  --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
  --train_caption_annotations_file="${TRAIN_CAPTION_ANNOTATIONS_FILE}" \
  --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
  --val_caption_annotations_file="${VAL_CAPTION_ANNOTATIONS_FILE}" \
  --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
  --output_dir="${OUTPUT_DIR}"

python tools/create_coco_vocab.py --logtostderr \
  --train_caption_annotations_file="${TRAIN_CAPTION_ANNOTATIONS_FILE}" \
  --glove_file="zoo/glove.6B.50d.txt" \
  --vocabulary_file="${OUTPUT_DIR}/coco_vocab.txt" \
  --vocabulary_weights_file="${OUTPUT_DIR}/coco_vocab_50d.npy" \
  --min_word_freq="20"

exit 0
