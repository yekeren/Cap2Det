#!/bin/sh

set -o errexit
set -o nounset
set -x

download() {
  local -r dir=$1
  local -r filename=$2
  local -r output_dir=$3

  if [ ! -f "${output_dir}/${filename}" ]; then
    wget -O "${output_dir}/${filename}" "${dir}/${filename}"
  fi
}

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 DIRECTORY"
  exit 1
fi

raw_data=$1
log="log"  # Directory to write log information.
model_zoo="zoo"  # Directory to save pre-trained models.

mkdir -p "${log}"
mkdir -p "${raw_data}"
mkdir -p "${model_zoo}"

# Download Flickr30K images and annotations.

image_dir="http://shannon.cs.illinois.edu/DenotationGraph/data"
images="flickr30k-images.tar"
annotations="flickr30k.tar.gz"

download "${image_dir}" "${images}" "${raw_data}"
download "${image_dir}" "${annotations}" "${raw_data}"
cd "${raw_data}"; tar xzf "${annotations}"; cd -

num_processes=10
proposal_data="${raw_data}/proposal_data"

if [ ! -d "${proposal_data}" ]; then
  for ((i=0;i<${num_processes};++i)); do
    python "dataset-tools/create_flickr30k_selective_search_data.py" \
      --alsologtostderr \
      --process_indicator="${i}/${num_processes}" \
      --image_tar_file="${raw_data}/${images}" \
      --output_dir="${proposal_data}" \
      > "${log}/flickr30k_proposal_${i}.log" 2>&1 &
  done
fi

python "dataset-tools/create_flickr30k_tf_record.py" \
  --alsologtostderr \
  --image_tar_file="${raw_data}/flickr30k-images.tar" \
  --proposal_data_path="${proposal_data}" \
  --annotation_path="${raw_data}/results_20130124.token" \
  --output_path="${raw_data}/flickr30k_trainval.record" \
  --number_of_parts=20










exit 0


mkdir -p ${OUTPUT_DIR}

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

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

