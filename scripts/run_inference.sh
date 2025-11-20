#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODE=${1:-profile}

if [ "$MODE" = "profile" ]; then
  echo "=========================================="
  echo "[Inference] Running all configurations"
  echo "=========================================="
  echo ""
  
  # Run keep09 configuration
  echo "[Config: keep09] Profiling baseline vs SDTP (keep_ratio=0.9)"
  echo "----------------------------------------"
  python3 -u src/inference_sdtp.py \
    --mode profile \
    --config keep09 \
    --lengths 1024 2048 4096 8192 16384 32768 \
    "${@:2}"
  echo ""
  
  # Run keep08 configuration
  echo "[Config: keep08] Profiling baseline vs SDTP (keep_ratio=0.8)"
  echo "----------------------------------------"
  python3 -u src/inference_sdtp.py \
    --mode profile \
    --config keep08 \
    --lengths 1024 2048 4096 8192 16384 32768 \
    "${@:2}"
  echo ""
  
  # Run keep07 configuration
  echo "[Config: keep07] Profiling baseline vs SDTP (keep_ratio=0.7)"
  echo "----------------------------------------"
  python3 -u src/inference_sdtp.py \
    --mode profile \
    --config keep07 \
    --lengths 1024 2048 4096 8192 16384 32768 \
    "${@:2}"
  echo ""
  
  echo "=========================================="
  echo "[OK] All configurations completed!"
  echo "=========================================="
  
elif [ "$MODE" = "generate" ]; then
  shift
  PROMPT="$*"
  if [ -z "$PROMPT" ]; then
    PROMPT="Hello, SDTP! Please introduce yourself."
  fi
  echo "[Inference] Generating text with baseline model"
  python3 -u src/inference_sdtp.py \
    --mode generate \
    --prompt "$PROMPT"
else
  echo "Unknown mode: $MODE"
  echo "Usage: $0 [profile|generate] [extra-args...]"
  echo ""
  echo "  profile (default): Run keep09, keep08, and keep07 configurations"
  echo "  generate: Generate text with the model"
  echo ""
  echo "Examples:"
  echo "  $0                    # Run all configurations"
  echo "  $0 profile           # Same as above"
  echo "  $0 generate 'Your prompt here'"
  exit 1
fi
