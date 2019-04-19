#!/bin/sh

set -x

TRAIN_IMAGE_FILE="raw_data/train2017.zip"
VAL_IMAGE_FILE="raw_data/val2017.zip"
TEST_IMAGE_FILE="raw_data/test2017.zip"
TRAIN_ANNOTATIONS_FILE="raw_data/annotations/instances_train2017.json"
TRAIN_CAPTION_ANNOTATIONS_FILE="raw_data/annotations/captions_train2017.json"
VAL_ANNOTATIONS_FILE="raw_data/annotations/instances_val2017.json"
VAL_CAPTION_ANNOTATIONS_FILE="raw_data/annotations/captions_val2017.json"
TESTDEV_ANNOTATIONS_FILE="raw_data/annotations/image_info_test-dev2017.json"
OUTPUT_DIR="output"

mkdir -p ${OUTPUT_DIR}

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"

#protoc object_detection/protos/*.proto --python_out=. || exit -1

#for ((i=0;i<20;++i)); do
#python tools/create_coco_edge_box_data.py --logtostderr \
#  --parts="${i}/20" \
#  --train_image_file="${TRAIN_IMAGE_FILE}" \
#  --val_image_file="${VAL_IMAGE_FILE}" \
#  --test_image_file="${TEST_IMAGE_FILE}" \
#  --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
#  --train_caption_annotations_file="${TRAIN_CAPTION_ANNOTATIONS_FILE}" \
#  --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
#  --val_caption_annotations_file="${VAL_CAPTION_ANNOTATIONS_FILE}" \
#  --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
#  --output_dir="raw_data/coco_egbox" \
#  >> "log/${i}.log" 2>&1 &
#done

#for ((i=0;i<20;++i)); do
#python tools/create_coco_selective_search_data.py --logtostderr \
#  --parts="${i}/20" \
#  --train_image_file="${TRAIN_IMAGE_FILE}" \
#  --val_image_file="${VAL_IMAGE_FILE}" \
#  --test_image_file="${TEST_IMAGE_FILE}" \
#  --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
#  --train_caption_annotations_file="${TRAIN_CAPTION_ANNOTATIONS_FILE}" \
#  --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
#  --val_caption_annotations_file="${VAL_CAPTION_ANNOTATIONS_FILE}" \
#  --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
#  --output_dir="raw_data/coco_ssbox_quality" \
#  >> "log/${i}.log" 2>&1 &
#done

#python tools/create_coco_tf_record.py --logtostderr \
#  --train_image_file="${TRAIN_IMAGE_FILE}" \
#  --val_image_file="${VAL_IMAGE_FILE}" \
#  --test_image_file="${TEST_IMAGE_FILE}" \
#  --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
#  --train_caption_annotations_file="${TRAIN_CAPTION_ANNOTATIONS_FILE}" \
#  --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
#  --val_caption_annotations_file="${VAL_CAPTION_ANNOTATIONS_FILE}" \
#  --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
#  --pseudo_labels_json_path="output/text_classification_mlp2.coco17.json" \
#  --output_dir="${OUTPUT_DIR}"
#
#exit 0

python tools/create_coco_vocab_tmp.py --logtostderr \
  --train_caption_annotations_file="${TRAIN_CAPTION_ANNOTATIONS_FILE}" \
  --glove_file="zoo/glove.6B.300d.txt" \
  --category_file="configs/coco_vocab.txt" \
  --vocabulary_file="configs/coco_open_vocab.txt.bak" \
  --vocabulary_weights_file="configs/coco_open_vocab_300d.npy.bak" \
  --min_word_freq="10"

exit 0
