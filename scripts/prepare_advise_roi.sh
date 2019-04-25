#!/bin/sh

set -x

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

#export CUDA_VISIBLE_DEVICES=0
#python tools/img_to_npy.py \
#  --image_data_path="raw_data/test_images" \
#  > log/img_to_npy_test.log 2>&1 &
#
#export CUDA_VISIBLE_DEVICES=1
#python tools/img_to_npy.py \
#  --image_data_path="raw_data/train_images" \
#  > log/img_to_npy_train.log 2>&1 &

#export CUDA_VISIBLE_DEVICES=0
#python tools/wsod_roi_to_npy.py \
#  --image_data_path="raw_data/ads_train_images" \
#  --bounding_box_json_path="raw_data/ads_wsod.json/trainval/" \
#  >> log/trainval.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0
python tools/wsod_roi_to_npy.py \
  --image_data_path="raw_data/ads_test_images" \
  --bounding_box_json_path="raw_data/ads_wsod.json/test/" \
  > log/test.log 2>&1 &

exit 0
