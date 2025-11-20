# SDTP: Selective Dynamic Token Pruning for Large Language Models [Part1]

This project implements a reproduction of [Saliency-driven Dynamic Token Pruning for Large Language Models](ref_paper.pdf) based on [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B).

## Project Structure

```
SDTP/
├── checkpoints/          # Model weights and checkpoints
│   ├── pruning_module.pt    # Stage 2 trained Token Pruner
│   ├── saliency.pt          # Stage 1 saliency baseline
│   └── qwen2-7b-instruct/   # [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) model weights
│       ├── config.json          # Model configuration
│       ├── model.safetensors    # Model weights (safetensors format)
│       ├── tokenizer.json       # Tokenizer configuration
│       ├── tokenizer.model      # Tokenizer model file
│       └── ...                  # Other model files (e.g., generation_config.json, etc.)
│
├── data/                 # Datasets
│   └── raw/                 # Raw data files (e.g., Dolly-15k)
│
├── results/              # Experimental results and reports
│   ├── fig/                 # Visualization figures
│   └── part1_sum.md         # Stage 1 summary report
│
├── scripts/              # Execution scripts
│   ├── run_stage1.sh        # Stage 1: Saliency computation
│   ├── run_stage2.sh        # Stage 2: Pruning module training
│   ├── run_inference.sh     # Single GPU inference
│   ├── run_inference_multigpu.sh  # Multi-GPU inference
│   ├── check_full_env.sh    # Environment check
│   └── install.sh           # Dependency installation
│
└── src/                  # Source code
    ├── stage1_saliency.py        # Stage 1: Gradient × hidden states
    ├── stage2_pruning.py         # Stage 2: Learnable Token Pruner
    ├── sdtp_model.py            # Core model with pruning logic
    ├── inference_sdtp.py        # Single GPU inference
    ├── inference_sdtp_multigpu.py  # Multi-GPU inference
    └── multigpu_test.py          # Multi-GPU memory profiling
```

## Quick Start

### Requirements

- Python 3.10+
- CUDA 12.1+
- **Hardware**: 8× NVIDIA RTX 5880 Ada Generation (48GB VRAM each)
  - Single GPU mode: Uses one GPU
  - Multi-GPU mode: Uses all 8 GPUs
- ≥50GB disk space for model storage

### Installation

**Prerequisites:**
- Install CUDA 12.1+ (includes nvcc compiler) and NVIDIA GPU drivers
- Verify CUDA installation: `nvcc --version` and `nvidia-smi`

**Install Python packages:**
```bash
pip install -r requirements.txt
```

**Note:** PyTorch will automatically use the installed CUDA version. For CUDA 12.x, install PyTorch with:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Usage

1. **Stage 1: Saliency Computation**
   ```bash
   bash scripts/run_stage1.sh
   ```

2. **Stage 2: Pruning Module Training**
   ```bash
   bash scripts/run_stage2.sh
   ```

3. **Inference**
   ```bash
   # Single GPU
   bash scripts/run_inference.sh
   
   # Multi-GPU
   bash scripts/run_inference_multigpu.sh
   ```

## Part 1 Summary

Part 1 implementation completed:
- ✅ Saliency baseline computation
- ✅ Token Pruner module training
- ✅ Single GPU inference (2.6-3× speedup)
- ✅ Multi-GPU inference (8-10× speedup)

See [Part 1 Summary Report](results/part1_sum.md) for detailed results.

## Key Features

- **Dynamic Token Pruning**: Remove redundant tokens during prefill to reduce computation
- **Layer-wise Pruning**: Progressive pruning across Transformer layers
- **Multi-GPU Support**: Automatic distributed inference for long sequences
- **Learnable Importance Predictor**: Lightweight MLP replaces heuristic methods

## Results

- **Single GPU** (NVIDIA RTX 5880 Ada, 48GB): 2.6-3.0× prefill speedup
- **Multi-GPU** (8× NVIDIA RTX 5880 Ada Generation, 48GB each): Up to 10× end-to-end speedup
- **Memory Savings**: Up to 34% GPU memory reduction
- **Performance**: Maintains comparable performance with 65% token pruning

