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

# Download MSCOCO images.

image_dir="http://images.cocodataset.org/zips"
train_images="train2017.zip"
val_images="val2017.zip"
test_images="test2017.zip"

download "${image_dir}" "${train_images}" "${raw_data}"
download "${image_dir}" "${val_images}" "${raw_data}"
download "${image_dir}" "${test_images}" "${raw_data}"

# Download MSCOCO annotations.

annotation_dir="http://images.cocodataset.org/annotations"
trainval_annotations="annotations_trainval2017.zip"
testdev_annotations="image_info_test2017.zip"

download "${annotation_dir}" "${trainval_annotations}" "${raw_data}"
download "${annotation_dir}" "${testdev_annotations}" "${raw_data}"
unzip -n "${raw_data}/${trainval_annotations}" -d "${raw_data}"
unzip -n "${raw_data}/${testdev_annotations}" -d "${raw_data}"

# Download GloVe models.

glove_dir="http://nlp.stanford.edu/data"
glove_file="glove.6B.zip"

download "${glove_dir}" "${glove_file}" "${model_zoo}"
unzip -n "${model_zoo}/${glove_file}" -d "${model_zoo}"

# Extract Selective Search proposals.

num_processes=50
proposal_data="${raw_data}/proposal_data"
if [ ! -d "${proposal_data}" ]; then
  for ((i=0;i<${num_processes};++i)); do
    python "dataset-tools/create_coco_selective_search_data.py" \
      --logtostderr \
      --process_indicator="${i}/${num_processes}" \
      --train_image_file="${raw_data}/${train_images}" \
      --val_image_file="${raw_data}/${val_images}" \
      --test_image_file="${raw_data}/${test_images}" \
      --train_annotations_file="${raw_data}/annotations/instances_train2017.json" \
      --val_annotations_file="${raw_data}/annotations/instances_val2017.json" \
      --testdev_annotations_file="${raw_data}/annotations/image_info_test-dev2017.json" \
      --output_dir="${proposal_data}" \
      > "${log}/coco17_proposal_${i}.log" 2>&1 &
  done
  wait
fi

# Convert MSCOCO dataset to tfrecord.

python "dataset-tools/create_coco_tf_record.py" \
  --logtostderr \
  --train_image_file="${raw_data}/${train_images}" \
  --val_image_file="${raw_data}/${val_images}" \
  --test_image_file="${raw_data}/${test_images}" \
  --train_annotations_file="${raw_data}/annotations/instances_train2017.json" \
  --train_caption_annotations_file="${raw_data}/annotations/captions_train2017.json" \
  --val_annotations_file="${raw_data}/annotations/instances_val2017.json" \
  --val_caption_annotations_file="${raw_data}/annotations/captions_val2017.json" \
  --testdev_annotations_file="${raw_data}/annotations/image_info_test-dev2017.json" \
  --proposal_data_path="${proposal_data}" \
  --output_dir="${raw_data}"

# Gather the vocabulary.

python "dataset-tools/create_coco_vocab.py" \
  --logtostderr \
  --train_caption_annotations_file="${raw_data}/annotations/captions_train2017.json" \
  --glove_file="${model_zoo}/glove.6B.300d.txt" \
  --output_vocabulary_file="${raw_data}/coco_open_vocab.txt" \
  --output_vocabulary_word_embedding_file="${raw_data}/coco_open_vocab_300d.npy" \
  --min_word_freq="10"
