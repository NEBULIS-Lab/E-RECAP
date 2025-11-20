#!/bin/bash
# Generate latency curves from JSON data
# Automatically processes both single-GPU and multi-GPU results if available
# Supports both old format (separate baseline/sdtp files) and new format (combined files)

OUT_DIR=${1:-"results/fig"}

echo "[Plot Latency] Processing available results..."

# Single-GPU results - try new format first, then fallback to old format
if [ -f "results/latency_results_keep09.json" ]; then
    echo "[Single-GPU keep09 Config] Generating curves..."
    python3 src/evaluation/plot_latency.py \
        --combined "results/latency_results_keep09.json" \
        --out_dir "$OUT_DIR" \
        --prefix "singlegpu_keep09"
    echo ""
fi

if [ -f "results/latency_results_keep08.json" ]; then
    echo "[Single-GPU keep08 Config] Generating curves..."
    python3 src/evaluation/plot_latency.py \
        --combined "results/latency_results_keep08.json" \
        --out_dir "$OUT_DIR" \
        --prefix "singlegpu_keep08"
    echo ""
fi

if [ -f "results/latency_results_keep07.json" ]; then
    echo "[Single-GPU keep07 Config] Generating curves..."
    python3 src/evaluation/plot_latency.py \
        --combined "results/latency_results_keep07.json" \
        --out_dir "$OUT_DIR" \
        --prefix "singlegpu_keep07"
    echo ""
fi

# Fallback to old format (backward compatibility)
if [ -f "results/latency_baseline.json" ] && [ -f "results/latency_sdtp.json" ]; then
    if [ ! -f "results/latency_results_keep07.json" ] && [ ! -f "results/latency_results_keep08.json" ] && [ ! -f "results/latency_results_keep09.json" ]; then
        echo "[Single-GPU] Generating curves (old format)..."
        python3 src/evaluation/plot_latency.py \
            --baseline "results/latency_baseline.json" \
            --sdtp "results/latency_sdtp.json" \
            --out_dir "$OUT_DIR" \
            --prefix "singlegpu"
        echo ""
    fi
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

