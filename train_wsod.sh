#!/bin/bash

set -o errexit
set -o nounset
set -x

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 MODEL_NAME"
  exit 1
fi

NAME=$1
MODEL_DIR="logs/${NAME}"
PIPELINE_CONFIG_PATH="configs/${NAME}.pbtxt"

if [ ! -f "${PIPELINE_CONFIG_PATH}" ]; then
  echo "Config file ${PIPELINE_CONFIG_PATH} does not exist."
  exit 1
fi


HOST0="127.0.0.1"

PS="${HOST0}:2220"
CHIEF="${HOST0}:2225"
WORKER0="${HOST0}:2226"
WORKER1="${HOST0}:2227"
WORKER2="${HOST0}:2228"
WORKER3="${HOST0}:2229"

# Evaluator.
export CUDA_VISIBLE_DEVICES=4
python "train/predict.py" \
  --alsologtostderr \
  --evaluator="pascal" \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --model_dir="${MODEL_DIR}" \
  --max_eval_examples=500 \
  --max_visl_examples=500 \
  --label_file="data/voc_label.txt" \
  --eval_log_dir="${MODEL_DIR}/eval_det" \
  --saved_ckpts_dir="${MODEL_DIR}/saved_ckpts" \
  --visl_file_path="${MODEL_DIR}/visl.html" \
  >> "log/${NAME}.eval.log" 2>&1 &

# Trainers.
server_list=(${PS} ${CHIEF} ${WORKER0} ${WORKER1} ${WORKER2})
CLUSTER="{\"chief\": [\"${CHIEF}\"], \"ps\": [\"${PS}\"], \"worker\": [\"${WORKER0}\", \"${WORKER1}\", \"${WORKER2}\"]}"

declare -A type_dict=(
  [${PS}]="ps"
  [${CHIEF}]="chief"
  [${WORKER0}]="worker"
  [${WORKER1}]="worker"
  [${WORKER2}]="worker"
  [${WORKER3}]="worker"
)
declare -A index_dict=(
  [${PS}]="0"
  [${CHIEF}]="0"
  [${WORKER0}]="0"
  [${WORKER1}]="1"
  [${WORKER2}]="2"
  [${WORKER3}]="3"
)
declare -A gpu_dict=(
  [${PS}]=""
  [${CHIEF}]="0"
  [${WORKER0}]="1"
  [${WORKER1}]="2"
  [${WORKER2}]="3"
  [${WORKER3}]="4"
)

for server in ${server_list[@]}; do
  gpu=${gpu_dict[${server}]}
  type=${type_dict[${server}]}
  index=${index_dict[${server}]}

  export CUDA_VISIBLE_DEVICES="${gpu}"
  export TF_CONFIG="{\"cluster\": ${CLUSTER}, \"task\": {\"type\": \"${type}\", \"index\": ${index}}}"  
  python "train/trainer_main.py" \
    --alsologtostderr \
    --type="${type}${index}" \
    --pipeline_proto=${PIPELINE_CONFIG_PATH} \
    --model_dir="${MODEL_DIR}" \
    >> "log/${NAME}.${type}${index}.log" 2>&1 &
done

exit 0
