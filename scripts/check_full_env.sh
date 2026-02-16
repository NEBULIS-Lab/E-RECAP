#!/usr/bin/env bash
set -e

echo "========================================"
echo "         E-RECAP Environment Check"
echo "========================================"
echo

# 0. Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
echo "[Project Root] $PROJECT_ROOT"
echo

# 1. GPU & Driver
echo "=== 1. GPU & Driver =================================="
nvidia-smi || echo "nvidia-smi NOT found"
echo

# 2. CUDA Toolkit (nvcc)
echo "=== 2. CUDA Toolkit (nvcc) ============================"
nvcc --version || echo "nvcc not installed"
echo

# 3. GPU Topology
echo "=== 3. GPU Topology ==================================="
nvidia-smi topo -m || echo "Cannot get topology"
echo

# 4. System toolchain
echo "=== 4. System Toolchain ================================"
for pkg in gcc g++ make git wget; do
  if command -v $pkg >/dev/null 2>&1; then
    echo "$pkg: OK ($(which $pkg))"
  else
    echo "$pkg: MISSING"
  fi
done
echo

# 5. Python & Pip
echo "=== 5. Python & Pip ===================================="
python3 --version
pip --version
echo

# 6. PyTorch CUDA
echo "=== 6. PyTorch CUDA Check =============================="
python3 - << 'EOF'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(0))
EOF
echo

# 7. FlashAttention
echo "=== 7. FlashAttention ================================"
python3 - << 'EOF'
try:
    import flash_attn
    print("flash_attn version:", flash_attn.__version__)
except Exception as e:
    print("FlashAttention FAILED:", e)
EOF
echo

# 8. xFormers
echo "=== 8. xFormers ========================================"
python3 - << 'EOF'
try:
    import xformers
    print("xformers version:", xformers.__version__)
except Exception as e:
    print("xformers FAILED:", e)
EOF
echo

# 9. Transformers & Tokenizers
echo "=== 9. Transformers & Tokenizers ======================"
python3 - << 'EOF'
try:
    import transformers, tokenizers
    print("transformers:", transformers.__version__)
    print("tokenizers:", tokenizers.__version__)
except Exception as e:
    print("FAILED:", e)
EOF
echo

# 10. lm-eval-harness (for 5-shot tasks)
echo "=== 10. LM-EVAL-HARNESS ==============================="
python3 - << 'EOF'
try:
    import lm_eval
    print("lm-eval imported successfully")
except Exception as e:
    print("lm-eval FAILED:", e)
EOF
echo

# 11. NCCL Library
echo "=== 11. NCCL Library =================================="
python3 - << 'EOF'
import torch
try:
    import torch.distributed as dist
    print("torch distributed backend available:", dist.is_available())
    print("NCCL backend available:", dist.is_nccl_available())
except Exception as e:
    print("NCCL FAILED:", e)
EOF
echo

# 12. Disk space
echo "=== 12. Disk Space ====================================="
df -h .
echo

# 13. Model directory check (project-local)
echo "=== 13. Model Directory (checkpoints/) =================="
if ls checkpoints/*/config.json >/dev/null 2>&1; then
  echo "Found at least one model under checkpoints/:"
  ls -1 checkpoints/*/config.json | sed 's|/config.json$||' | sed 's|^|  - |'
else
  echo "No model found under checkpoints/<model-name>/config.json"
  echo "Tip: place a local HF model directory under checkpoints/ and pass --model_path to scripts."
fi
echo

echo "========================================"
echo "        Environment Check Completed"
echo "========================================"
