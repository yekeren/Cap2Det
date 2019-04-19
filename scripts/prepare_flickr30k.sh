#!/bin/sh

set -x


mkdir -p ${OUTPUT_DIR}

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

#for ((i=0;i<20;++i)); do
#  python tools/create_flickr30k_selective_search_data.py \
#    --alsologtostderr \
#    --parts="$i/20" \
#    --image_tar_file="raw_data/flickr30k-images.tar" \
#    --output_path="raw_data/flickr30k_ssbox_quality/" \
#    >> "log/${i}.log" 2>&1 &
#done

#python tools/create_flickr30k_tf_record.py \
#  --alsologtostderr \
#  --number_of_parts=20 \
#  --image_tar_file="raw_data/flickr30k-images.tar" \
#  --proposal_data="raw_data/flickr30k_ssbox_quality/" \
#  --annotation_path="raw_data/flickr30k-captions.token" \
#  --pseudo_labels_json_path="output/text_classification_mlp2.flickr30k.json" \
#  --output_path="output/flickr30k_trainval_with_pseudo.record" \
#  || exit -1

python tools/create_flickr30k_vocab_tmp.py \
  --alsologtostderr \
  --annotation_path="raw_data/flickr30k-captions.token" \
  --output_path="output/flickr30k_trainval_ssbox_quality" \
  --glove_file="zoo/glove.6B.300d.txt" \
  --category_file="configs/coco_vocab.txt" \
  --vocabulary_file="configs/flickr30k_open_vocab.txt.bak" \
  --vocabulary_weights_file="configs/flickr30k_open_vocab_300d.npy.bak" \
  --min_word_freq="10" \
  || exit -1
echo done
exit 0

