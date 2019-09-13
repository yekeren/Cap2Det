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

# Download GloVe models.

glove_dir="http://nlp.stanford.edu/data"
glove_file="glove.6B.zip"

download "${glove_dir}" "${glove_file}" "${model_zoo}"
unzip -n "${model_zoo}/${glove_file}" -d "${model_zoo}"


# Convert Flickr30K dataset to tfrecord.

python "dataset-tools/create_flickr30k_tf_record.py" \
  --alsologtostderr \
  --image_tar_file="${raw_data}/flickr30k-images.tar" \
  --proposal_data_path="${proposal_data}" \
  --annotation_path="${raw_data}/results_20130124.token" \
  --output_path="${raw_data}/flickr30k_trainval.record" \
  --number_of_parts=20

# Gather the vocabulary.

python "dataset-tools/create_flickr30k_vocab.py" \
  --alsologtostderr \
  --annotation_path="${raw_data}/results_20130124.token" \
  --glove_file="${model_zoo}/glove.6B.300d.txt" \
  --output_vocabulary_file="${raw_data}/flickr30k_open_vocab.txt" \
  --output_vocabulary_word_embedding_file="${raw_data}/flickr30k_open_vocab_300d.npy" \
  --min_word_freq="10" \
