#!/bin/sh

set -x

#python tools/create_advise_tfrecord.py \
#  --number_of_parts=10 \
#  --image_data_path="raw_data/ads_train_images" \
#  --roi_feature_npy_path="raw_data/wsod.roi.npy" \
#  --tfrecord_output_path="output.advise/wsod_trainval.record" \
#  > log/train_tfrecord.log 2>&1 &

python tools/create_advise_tfrecord.py \
  --number_of_parts=10 \
  --image_data_path="raw_data/ads_test_images" \
  --roi_feature_npy_path="raw_data/wsod.roi.npy" \
  --tfrecord_output_path="output.advise/wsod_test.record" \
  > log/test_tfrecord.log 2>&1 &

exit 0
