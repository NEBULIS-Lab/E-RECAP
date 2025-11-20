#!/bin/bash
# lm-eval-harness runner script

TYPE=${1:-"baseline"}

echo "[lm-eval] Type: $TYPE"

python3 src/evaluation/lmeval_runner.py \
    --type "$TYPE" \
    --out "results/lmeval_${TYPE}.json"

