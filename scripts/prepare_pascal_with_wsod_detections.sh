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
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

year="VOC2012"
set_list=(
  trainval
)

#for set in ${set_list[@]}; do
#for ((i=0;i<20;++i)); do
#python tools/create_pascal_selective_search_data.py --alsologtostderr \
#  --data_dir="/afs/cs.pitt.edu/usr0/key36/DATASET/VOC/VOCdevkit" \
#  --set="${set}" \
#  --annotations_dir="Annotations" \
#  --year="${year}" \
#  --parts="${i}/20" \
#  --output_data_path="raw_data/voc_ssbox_quality/" \
#  --label_map_path="configs/pascal_label_map.pbtxt" \
#  --ignore_difficult_instances \
#  >> "log/${i}.log" 2>&1 &
#done
#done


for set in ${set_list[@]}; do
python tools/create_pascal_tf_record_with_wsod_detections.py --alsologtostderr \
  --data_dir="/afs/cs.pitt.edu/usr0/key36/DATASET/VOC/VOCdevkit" \
  --set="${set}" \
  --annotations_dir="Annotations" \
  --year="${year}" \
  --output_path="output/${year}_${set}_ssbox_quality_with_wsod.record" \
  --label_map_path="configs/pascal_label_map.pbtxt" \
  --ignore_difficult_instances \
  --wsod_detection_dir="voc2012.trainval.results/per_class_ssquality_12/" >> "mz_log/pascal_with_wsod.log" 2>&1 &
done

exit 0
