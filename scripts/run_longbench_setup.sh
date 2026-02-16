#!/bin/bash
# LongBench evaluation setup script (no inference execution)

TASK=${1:-"data/LongBench/narrativeqa.json"}
MODEL=${2:-"checkpoints/model"}
PRUNING_MODULE=${3:-"checkpoints/pruning_module.pt"}
OUTPUT=${4:-"results/longbench_setup.json"}

echo "[LongBench Setup] Preparing evaluation framework..."
echo "  Task: $TASK"
echo "  Model: $MODEL"
echo "  Pruning Module: $PRUNING_MODULE"
echo "  Output: $OUTPUT"

python3 src/evaluation/longbench/run_longbench.py \
    --task "$TASK" \
    --model "$MODEL" \
    --pruning_module "$PRUNING_MODULE" \
    --output "$OUTPUT"
