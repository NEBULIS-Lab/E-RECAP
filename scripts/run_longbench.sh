#!/bin/bash
# LongBench evaluation script

TASK=${1:-"hotpotqa"}
TYPE=${2:-"baseline"}
NUM_SAMPLES=${3:-30}

echo "[LongBench] Task: $TASK, Type: $TYPE, Samples: $NUM_SAMPLES"

python3 src/evaluation/longbench_eval.py \
    --task "$TASK" \
    --type "$TYPE" \
    --out "results/longbench_${TASK}_${TYPE}.json" \
    --num_samples "$NUM_SAMPLES"

