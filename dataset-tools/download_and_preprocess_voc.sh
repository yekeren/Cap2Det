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

mkdir -p "${log}"
mkdir -p "${raw_data}"

# Download VOC images and annotations.

voc12_dir="http://host.robots.ox.ac.uk/pascal/VOC/voc2012"
voc12_trainval_images="VOCtrainval_11-May-2012.tar"
voc12_trainval_annotations="VOCdevkit_18-May-2011.tar"

download "${voc12_dir}" "${voc12_trainval_images}" "${raw_data}"
download "${voc12_dir}" "${voc12_trainval_annotations}" "${raw_data}"

voc12_test_dir="http://pjreddie.com/media/files"
voc12_test_images="VOC2012test.tar"

download "${voc12_test_dir}" "${voc12_test_images}" "${raw_data}"

voc07_dir="http://host.robots.ox.ac.uk/pascal/VOC/voc2007"
voc07_trainval_images="VOCtrainval_06-Nov-2007.tar"
voc07_trainval_annotations="VOCdevkit_08-Jun-2007.tar"
voc07_test_images_with_annotations="VOCtest_06-Nov-2007.tar"

download "${voc07_dir}" "${voc07_trainval_images}" "${raw_data}"
download "${voc07_dir}" "${voc07_trainval_annotations}" "${raw_data}"
download "${voc07_dir}" "${voc07_test_images_with_annotations}" "${raw_data}"

if [ ! -d "${raw_data}/VOCdevkit" ]; then
  tar -C "${raw_data}" -xf "${raw_data}/${voc12_trainval_images}"
  tar -C "${raw_data}" -xf "${raw_data}/${voc12_trainval_annotations}"
  tar -C "${raw_data}" -xf "${raw_data}/${voc12_test_images}"

  tar -C "${raw_data}" -xf "${raw_data}/${voc07_trainval_images}"
  tar -C "${raw_data}" -xf "${raw_data}/${voc07_trainval_annotations}"
  tar -C "${raw_data}" -xf "${raw_data}/${voc07_test_images_with_annotations}"
fi

# Extract Selective Search proposals.

YEARS=(VOC2007 VOC2012)
SETS=(trainval test)

num_processes=10
proposal_data="${raw_data}/proposal_data"

if [ ! -d "${proposal_data}" ]; then
  for year in ${YEARS[@]}; do
    for set in ${SETS[@]}; do
      for ((i=0;i<${num_processes};++i)); do
        python "dataset-tools/create_pascal_selective_search_data.py" \
          --alsologtostderr \
          --data_dir="${raw_data}/VOCdevkit" \
          --year="${year}" \
          --set="${set}" \
          --annotations_dir="Annotations" \
          --process_indicator="${i}/${num_processes}" \
          --output_dir="${proposal_data}" \
          > "${log}/${year}_${set}_proposal_${i}.log" 2>&1 &
      done
      wait
    done
  done
fi

# Convert Pascal VOC dataset to tfrecord. 

for year in ${YEARS[@]}; do
  for set in ${SETS[@]}; do
    python "dataset-tools/create_pascal_tf_record.py" \
      --alsologtostderr \
      --data_dir="${raw_data}/VOCdevkit" \
      --year="${year}" \
      --set="${set}" \
      --annotations_dir="Annotations" \
      --num_shards=10 \
      --output_path="${raw_data}/${year}_${set}.record" \
      --label_map_path="data/pascal_label_map.pbtxt" \
      --proposal_data_path="${proposal_data}" \
      --ignore_difficult_instances
  done
done
