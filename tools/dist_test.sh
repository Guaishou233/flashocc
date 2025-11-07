#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
GPU_IDS=${GPU_IDS:-""}  # 新增：指定GPU ID
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 设置GPU可见性
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
