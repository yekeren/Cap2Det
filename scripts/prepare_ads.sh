#!/bin/sh

set -x


mkdir -p ${OUTPUT_DIR}

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=9

#for ((i=0;i<20;++i)); do
#  python tools/create_ads_selective_search_data.py \
#    --alsologtostderr \
#    --parts="$i/20" \
#    --image_data_dir="raw_data/ads_train_images" \
#    --output_path="raw_data/ads_train_ssbox_quality/" \
#    >> "log/${i}.log" 2>&1 &
#done
#
#for ((i=0;i<20;++i)); do
#  python tools/create_ads_selective_search_data.py \
#    --alsologtostderr \
#    --parts="$i/20" \
#    --image_data_dir="raw_data/ads_test_images" \
#    --output_path="raw_data/ads_test_ssbox_quality/" \
#    >> "log/${i}.log" 2>&1 &
#done

python tools/create_ads_tf_record.py \
  --alsologtostderr \
  --number_of_parts=20 \
  --image_data_dir="raw_data/ads_train_images" \
  --proposal_data="raw_data/ads_train_ssbox_quality/" \
  --annotation_path="raw_data/qa.json/" \
  --output_path="output/ads_trainval_ssbox_quality.record" \
  || exit -1

#python tools/create_ads_vocab.py \
#  --alsologtostderr \
#  --category_file="configs/ads.vocab" \
#  --image_data_dir="raw_data/ads_train_images" \
#  --annotation_path="raw_data/qa.json/" \
#  --glove_file="zoo/glove.6B.300d.txt" \
#  --vocabulary_file="configs/ads_open_vocab.vocab" \
#  --vocabulary_weights_file="configs/ads_open_vocab_300d.npy" \
#  --min_word_freq="10" \
#  || exit -1

echo done
exit 0

