#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODE=${1:-profile}
SHIFTED_ARGS=("${@:2}")

if [ "$MODE" = "profile" ]; then
  echo "[Inference] Profiling baseline vs SDTP"
  python -u src/inference_sdtp.py \
    --mode profile \
    --lengths 4096 8192 16384 32768 \
    "${SHIFTED_ARGS[@]}"
elif [ "$MODE" = "generate" ]; then
  shift
  PROMPT="$*"
  if [ -z "$PROMPT" ]; then
    PROMPT="Hello, SDTP! Please introduce yourself."
  fi
  echo "[Inference] Generating text with baseline model"
  python -u src/inference_sdtp.py \
    --mode generate \
    --prompt "$PROMPT"
else
  echo "Unknown mode: $MODE"
  echo "Usage: $0 [profile|generate] [prompt...]"
  exit 1
fi
