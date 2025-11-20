#!/bin/bash
# Generate latency curves from JSON data
# Automatically processes both single-GPU and multi-GPU results if available

OUT_DIR=${1:-"results/fig"}

echo "[Plot Latency] Processing available results..."

# Single-GPU results
if [ -f "results/latency_baseline.json" ] && [ -f "results/latency_sdtp.json" ]; then
    echo "[Single-GPU] Generating curves..."
    python3 src/evaluation/plot_latency.py \
        --baseline "results/latency_baseline.json" \
        --sdtp "results/latency_sdtp.json" \
        --out_dir "$OUT_DIR" \
        --prefix "singlegpu"
    echo ""
fi

# Multi-GPU results
if [ -f "results/latency_baseline_multigpu.json" ] && [ -f "results/latency_sdtp_multigpu.json" ]; then
    echo "[Multi-GPU] Generating curves..."
    python3 src/evaluation/plot_latency.py \
        --baseline "results/latency_baseline_multigpu.json" \
        --sdtp "results/latency_sdtp_multigpu.json" \
        --out_dir "$OUT_DIR" \
        --prefix "multigpu"
    echo ""
fi

echo "[OK] All available plots generated!"

