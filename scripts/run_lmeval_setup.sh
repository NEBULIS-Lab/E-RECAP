#!/usr/bin/env bash
# lm-eval-harness setup script for SDTP (no inference execution)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default arguments
TASK_CONFIG="${1:-data/LongBench/narrativeqa.json}"
MODEL_TYPE="${2:-baseline}"  # "baseline" or "sdtp"
OUTPUT_DIR="${3:-results}"

# Determine pruning module path based on model type
if [ "$MODEL_TYPE" == "sdtp" ]; then
    PRUNER="checkpoints/pruning_module.pt"
else
    PRUNER=""
fi

# Auto-generate output filename
TASK_NAME=$(basename "$TASK_CONFIG" .json)
OUTPUT_FILE="$OUTPUT_DIR/lmeval_${TASK_NAME}_${MODEL_TYPE}_setup.json"

echo "=========================================="
echo "[LM-EVAL] SDTP Evaluation Setup"
echo "=========================================="
echo "Task config: $TASK_CONFIG"
echo "Model type: $MODEL_TYPE"
echo "Pruning module: ${PRUNER:-None (baseline)}"
echo "Output: $OUTPUT_FILE"
echo "=========================================="

# Check if task file exists (optional, script will continue if not)
if [ ! -f "$TASK_CONFIG" ]; then
    echo "[Warning] Task file not found: $TASK_CONFIG"
    echo "[Info] This is OK in setup phase - file will be created later"
    echo "[Info] Continuing with setup validation..."
fi

# Run setup script (no inference)
python3 src/evaluation/lmeval/run_lmeval.py \
    --task_config "$TASK_CONFIG" \
    --model_name "checkpoints/qwen2-7b-instruct" \
    --pruner "${PRUNER}" \
    --output "$OUTPUT_FILE" \
    --device "cuda"

echo ""
echo "[OK] Setup completed: $OUTPUT_FILE"

