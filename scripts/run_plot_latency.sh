#!/bin/bash
# Generate latency curves from JSON data

BASELINE=${1:-"results/latency_baseline.json"}
SDTP=${2:-"results/latency_sdtp.json"}
OUT_DIR=${3:-"results/fig"}

echo "[Plot Latency] Generating curves..."
echo "  Baseline: $BASELINE"
echo "  SDTP: $SDTP"
echo "  Output: $OUT_DIR"

python3 src/evaluation/plot_latency.py \
    --baseline "$BASELINE" \
    --sdtp "$SDTP" \
    --out_dir "$OUT_DIR"

