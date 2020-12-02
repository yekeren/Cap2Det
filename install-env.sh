#!/bin/sh

set -o errexit
set -o nounset
set -x


# Checkout the cap2det branch of the `tensorflow/models`.
if [ ! -d "tensorflow_models" ]; then
  git clone "https://github.com/yekeren/tensorflow_models.git"
fi

cd "tensorflow_models" && git checkout cap2det
cd -

protoc protos/*.proto --python_out=.
cd "tensorflow_models/research/"
protoc object_detection/protos/*.proto --python_out=.
cd -

# Download the ImageNet pre-trained classification model.
cd zoo
model_dir="inception_v2_2016_08_28"
if [ ! -d "${model_dir}" ]; then
  mkdir ${model_dir}
  cd ${model_dir} && wget "http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz"
  tar xzvf "inception_v2_2016_08_28.tar.gz"
  rm "inception_v2_2016_08_28.tar.gz"
fi
