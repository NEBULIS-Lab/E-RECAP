#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Installing Python dependencies ==="

pip install --upgrade pip

# Install PyTorch (CUDA 12.1 wheel)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install general requirements
pip install -r requirements.txt

echo "=== Installation complete ==="