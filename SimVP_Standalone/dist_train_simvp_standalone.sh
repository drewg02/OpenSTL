#!/usr/bin/env bash

GPUS=$1
PY_ARGS=${@:2}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CMD="torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    --master_addr=$MASTER_ADDR \
    SimVP_Standalone/train_simvp_standalone.py --dist \
    --launcher=\"pytorch\" ${PY_ARGS}"

echo "Running command: $CMD"
eval $CMD