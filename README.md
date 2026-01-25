<div align="center">

<h1>E-RECAP: Embodied REplanning with Cost-Aware Pruning</h1>

</div>

![](https://img.shields.io/github/last-commit/NEBULIS-Lab/E-RECAP?color=green) ![](https://img.shields.io/badge/version-2.0-blue) <a href="https://nebulis-lab.com/"><img src="https://img.shields.io/badge/NEBULIS%20Lab-Website-6366F1.svg" alt="NEBULIS Lab"></a> <a href="#"><img src="https://img.shields.io/badge/arXiv-coming%20soon-009688.svg" alt="arXiv"></a> <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> <a href="https://huggingface.co/docs/transformers"><img alt="Transformers" src="https://img.shields.io/badge/Transformers-ffd21e?logo=huggingface&logoColor=black"></a> <a href="https://www.deepspeed.ai/"><img alt="DeepSpeed" src="https://img.shields.io/badge/DeepSpeed-0A66FF?logo=microsoft&logoColor=white"></a>

This project implements E-RECAP (Embodied REplanning with Cost-Aware Pruning), a system-level, drop-in method for accelerating replanning in embodied agents by cost-aware pruning of planner context. E-RECAP operates as a Planner optimization module that can be seamlessly integrated into embodied AI systems without modifying task definitions, environments, or control policies.

## Overview

In embodied AI systems, agents frequently need to replan due to partial observability, dynamic environments, and execution uncertainties. When using LLM/VLM as high-level planners, each replanning cycle requires processing long contexts that accumulate over time, making replanning a major computational bottleneck—especially in multi-agent settings where context grows with the number of agents.

E-RECAP addresses this by:
- **Learning task-agnostic token importance** from large-scale instruction-following data ([Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [Self-Instruct](https://arxiv.org/abs/2212.10560))
- **Cost-aware dynamic pruning** of planner context during replanning, reducing computation while preserving decision quality
- **System-level integration** that works with any Transformer-based planner without modifying perception or control modules

E-RECAP is evaluated in both single-agent and cooperative multi-agent settings, with embodied evaluation planned on [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) ([PointNav](https://aihabitat.org/docs/habitat-lab/habitat_task.html#pointnav)/[ObjectNav](https://aihabitat.org/docs/habitat-lab/habitat_task.html#objectnav) tasks).

## Project Structure

```
E-RECAP/
├── checkpoints/          # Model weights and checkpoints
│   ├── pruning_module.pt    # Stage 2 trained Token Pruner (required for inference)
│   ├── saliency.pt          # Stage 1 saliency baseline (optional)
│   └── <model-name>/        # Your model directory (e.g., qwen2-7b-instruct, llama2-7b, etc.)
│       ├── config.json          # Model configuration (required)
│       ├── model.safetensors    # Model weights (or model.bin, required)
│       ├── tokenizer.json       # Tokenizer configuration (required)
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
    ├── erecap_model.py            # Core model with pruning logic
    ├── inference_erecap.py        # Single GPU inference
    ├── inference_erecap_multigpu.py  # Multi-GPU inference
    ├── multigpu_test.py          # Multi-GPU memory profiling
    └── multi_agent/              # Cooperative multi-agent planning
        ├── cooperative_planner.py    # Main planner with E-RECAP integration
        ├── context_buffer.py         # Shared planning context buffer
        ├── structured_output.py      # Structured agent output parser
        ├── agent_config.py           # Agent configuration definitions
        ├── task_definitions.py       # Task step definitions
        ├── framework_wrapper.py      # Optional CrewAI/LangChain support
        └── framework_optional/       # Optional framework files (not in Git)
            └── agents_config.json    # Agent config for CrewAI (if used)
```

## Quick Start

### Requirements

- Python 3.10+
- CUDA 12.1+
- ≥50GB disk space for model storage

### Hardware Requirements

**Note:** The hardware requirements depend on the model you choose to use. The following are our test configurations, but you can run E-RECAP on any hardware that meets the minimum requirements for your selected model.

**Our test setup:**
- 8× NVIDIA RTX 5880 Ada Generation (48GB VRAM each)
  - Single GPU mode: Uses one GPU
  - Multi-GPU mode: Uses all 8 GPUs

**Recommended VRAM by model:**

<table style="font-size: 0.85em;">
<thead>
<tr>
<th style="text-align:left">Model</th>
<th>Params</th>
<th>Rec VRAM</th>
<th style="text-align:left">Model</th>
<th>Params</th>
<th>Rec VRAM</th>
</tr>
</thead>
<tbody>
<tr><td style="text-align:left"><a href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf">LLaMA-2</a></td><td>7B / 13B</td><td>~14 GB / ~26 GB</td><td style="text-align:left"><a href="https://huggingface.co/meta-llama/Llama-3-8b">LLaMA-3 / 3.1</a></td><td>8B / 70B</td><td>~16 GB / ~140 GB</td></tr>
<tr><td style="text-align:left"><a href="https://huggingface.co/meta-llama/Llama-3.2-11b">LLaMA-3.2</a></td><td>1B / 3B / 11B / 90B</td><td>~2 GB / ~6 GB / ~22 GB / ~180 GB</td><td style="text-align:left"><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">Mistral</a></td><td>7B</td><td>8 GB</td></tr>
<tr><td style="text-align:left"><a href="https://huggingface.co/Qwen/Qwen2-14B">Qwen2</a></td><td>7B / 14B / 32B / 72B</td><td>~14 GB / ~28 GB / ~64 GB / ~144 GB</td><td style="text-align:left"><a href="https://huggingface.co/Qwen/Qwen2.5-14B">Qwen2.5</a></td><td>7B / 14B</td><td>12 GB / 16 GB</td></tr>
<tr><td style="text-align:left"><a href="https://huggingface.co/Qwen/Qwen3-8B-Base">Qwen3</a></td><td>0.6B / 1.7B / 4B / 8B / 14B / 32B</td><td>~1.2 GB / ~3.4 GB / ~8 GB / ~16 GB / ~28 GB / ~64 GB</td><td style="text-align:left"><a href="https://huggingface.co/tomhao/yi_6b_chat_tool_use">Yi</a></td><td>6B / 13B</td><td>8 GB / 16 GB</td></tr>
<tr><td style="text-align:left"><a href="https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat">DeepSeek-LLM</a></td><td>7B / 67B</td><td>~14 GB / ~134 GB</td><td style="text-align:left"><a href="https://huggingface.co/gemma-2/9B">Gemma-2</a></td><td>9B</td><td>~18 GB</td></tr>
<tr><td style="text-align:left"><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct">Phi-3</a></td><td>3.8B / 7B</td><td>~7.6 GB / ~14 GB</td><td style="text-align:left"><a href="https://huggingface.co/THUDM/chatglm3-6b">ChatGLM3</a></td><td>6B</td><td>8 GB</td></tr>
<tr><td style="text-align:left"><a href="https://huggingface.co/baichuan-inc/Baichuan2-7B">Baichuan2</a></td><td>7B / 13B</td><td>~14 GB / ~26 GB</td><td style="text-align:left"><a href="https://huggingface.co/internlm/internlm2-20b">InternLM2</a></td><td>7B / 20B</td><td>12 GB / 24 GB</td></tr>
</tbody>
</table>

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

### File Organization

**Required files and their locations:**

1. **Model files** → `checkpoints/<model-name>/`
   - Place your HuggingFace-compatible model here
   - Must include: `config.json`, model weights (`.safetensors` or `.bin`), tokenizer files
   - Example structure:
     ```
     checkpoints/
     └── your-model-name/
         ├── config.json
         ├── model.safetensors (or model-*.safetensors)
         ├── tokenizer.json
         └── ...
     ```

2. **Pruning module** → `checkpoints/pruning_module.pt`
   - Generated by Stage 2 training
   - Model-specific (tied to model's `hidden_size`)
   - Required for inference

3. **Saliency baseline** → `checkpoints/saliency.pt`
   - Generated by Stage 1 (optional)
   - Used for training pruning module in Stage 2

4. **Training data** → `data/raw/dolly15k/` or `dolly15k/`
   - **Primary training data**: 
     - [Dolly-15k](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) ([HuggingFace](https://huggingface.co/datasets/databricks/databricks-dolly-15k)), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) ([HuggingFace](https://huggingface.co/datasets/tatsu-lab/alpaca)), processed [Self-Instruct](https://arxiv.org/abs/2212.10560) ([HuggingFace](https://huggingface.co/datasets/self_instruct))
     - Used for Stage 1 (saliency computation) and Stage 2 (pruning module training)
     - Learn task-agnostic token importance priors across diverse reasoning patterns
   - **Optional auxiliary data**: Textualized embodied samples from [ALFRED](https://arxiv.org/abs/1912.01734), [TEACh](https://arxiv.org/abs/2110.00534), [BabyAI](https://arxiv.org/abs/1810.08272), [BEHAVIOR-1K](https://arxiv.org/abs/2203.04051), [ProcTHOR](https://arxiv.org/abs/2206.06994)
     - Used at lower frequency to refine replanning-aware saliency patterns
     - Not required for E-RECAP to work, but helps improve replanning sensitivity
   - Can use any HuggingFace-compatible dataset

5. **Results** → `results/`
   - All benchmark results and logs are saved here

### Usage

#### Model Setup

**Note:** This repository provides Qwen2-7B-Instruct as an example for running and testing E-RECAP. The required files (`checkpoints/qwen2-7b-instruct/`, `checkpoints/pruning_module.pt`, and `checkpoints/saliency.pt`) are included. However, E-RECAP supports any HuggingFace-compatible Transformer model - the pruning module is model-agnostic and works with any model architecture that has a `hidden_size` configuration.

**Place your model files:**
1. Download or copy your model to `checkpoints/<model-name>/` directory
   - The model directory should contain `config.json`, model weights (`.safetensors` or `.bin`), and tokenizer files
   - Example: `checkpoints/qwen2-7b-instruct/`, `checkpoints/llama2-7b/`, etc.

2. **Configure model path** in `src/inference_erecap.py`:
   ```python
   MODEL_PATH = "checkpoints/<your-model-name>"
   ```
   Replace `<your-model-name>` with your actual model directory name.

#### Pre-flight Check

Verify that model and checkpoints exist (run from project root):
```bash
# Check model (replace <model-name> with your actual model directory)
ls -lh checkpoints/<model-name>/config.json

# Check checkpoints
ls -lh checkpoints/pruning_module.pt checkpoints/saliency.pt
```

If model or checkpoints are missing:
- **Model**: Download/copy your model to `checkpoints/<model-name>/`
- **Pruning module**: Run Stage 2 to train (see below)
- **Saliency**: Run Stage 1 to generate (optional)

#### Quick Verification

Test that all components are ready:
```bash
python3 -c "
import sys
sys.path.insert(0, 'src')
from inference_erecap import load_model_and_pruners, MODEL_PATH, PRUNING_CKPT
import os
print('✓ MODEL_PATH:', MODEL_PATH)
print('✓ MODEL_PATH exists:', os.path.exists(MODEL_PATH))
print('✓ PRUNING_CKPT exists:', os.path.exists(PRUNING_CKPT))
print('✓ All checks passed!')
"
```

**Note:** If `MODEL_PATH` doesn't exist, edit `src/inference_erecap.py` and set `MODEL_PATH` to your model directory path.

#### Stage 1: Saliency Computation (Optional)

Only needed if `checkpoints/saliency.pt` doesn't exist:
```bash
bash scripts/run_stage1.sh 1000
```

**Note:** This stage uses Dolly-15k dataset for training and the model specified in `src/stage1_saliency.py`. Make sure to set the correct model path there if using a different model.

#### Stage 2: Pruning Module Training (Required)

Only needed if `checkpoints/pruning_module.pt` doesn't exist:
```bash
bash scripts/run_stage2.sh 1e-4 2
```

Parameters:
- First argument: Learning rate (default: 1e-4)
- Second argument: Number of epochs (default: 2)

**Note:** This stage trains a model-specific pruning module using Dolly-15k dataset. The trained `pruning_module.pt` is tied to the model's `hidden_size`. If you change models, you may need to retrain the pruning module if the new model has a different `hidden_size`.

#### Inference

**Single GPU Inference:**

1. **Prefill-only benchmark** (fast, ~5-10 minutes):
   ```bash
   bash scripts/run_inference.sh profile prefill
   ```

2. **End-to-end benchmark** (includes decode, ~15-30 minutes):
   ```bash
   bash scripts/run_inference.sh profile end2end
   ```

3. **Text generation test** (quick verification):
   ```bash
   bash scripts/run_inference.sh generate "Hello, E-RECAP!"
   ```

**Multi-GPU Inference** (for long sequences, 32K+ tokens):
```bash
# Multi-GPU profiling
bash scripts/run_inference_multigpu.sh profile

# Multi-GPU text generation
bash scripts/run_inference_multigpu.sh generate "Your prompt here"
```

**Cooperative Multi-Agent Planning** (with E-RECAP context pruning):
```bash
# Run E-RECAP version (default: 15-step iterative replanning task)
bash scripts/run_cooperative_replanning.sh --keep_ratio 0.7 --save_results

# Run 10 times for longer evaluation (ensures >5 minutes runtime)
bash scripts/run_cooperative_replanning.sh --keep_ratio 0.7 --num_runs 10 --save_results

# Run baseline (no pruning) for comparison
bash scripts/run_cooperative_replanning.sh --baseline --save_results

# Run baseline 10 times
bash scripts/run_cooperative_replanning.sh --baseline --num_runs 10 --save_results

# Compare results (for 10 runs)
python3 src/multi_agent/compare_baseline_erecap.py \
    --baseline_file results/cooperative_planning_iterative_replanning_baseline_10runs.json \
    --erecap_file results/cooperative_planning_iterative_replanning_0.7_10runs.json
```

**Run single configuration directly:**
```bash
cd src
python3 -u inference_erecap.py \
  --mode profile \
  --config keep07 \
  --benchmark_mode prefill \
  --lengths 1024 2048 4096
```

#### Results Location

- **Single GPU results**: `results/latency_results_keep*.json`
- **Multi-GPU results**: `results/latency_erecap_multigpu.json`
- **Baseline results**: `results/latency_baseline_keep*.json`

### Available Scripts

The `scripts/` directory contains helper scripts for common tasks:

#### Core Scripts

- **`run_inference.sh`**: Single-GPU inference and benchmarking
  - `bash scripts/run_inference.sh profile prefill` - Prefill-only benchmark
  - `bash scripts/run_inference.sh profile end2end` - End-to-end benchmark
  - `bash scripts/run_inference.sh generate "prompt"` - Text generation

- **`run_inference_multigpu.sh`**: Multi-GPU inference for long sequences
  - `bash scripts/run_inference_multigpu.sh profile` - Multi-GPU profiling
  - `bash scripts/run_inference_multigpu.sh generate "prompt"` - Multi-GPU generation

- **`run_stage1.sh`**: Generate saliency baseline (optional)
  - `bash scripts/run_stage1.sh [num_samples]` - Default: 1000 samples

- **`run_stage2.sh`**: Train pruning module (required if missing)
  - `bash scripts/run_stage2.sh [learning_rate] [epochs]` - Default: 1e-4, 2 epochs

#### Utility Scripts

- **`install.sh`**: Install Python dependencies and PyTorch
  - `bash scripts/install.sh`

- **`check_full_env.sh`**: Comprehensive environment check
  - `bash scripts/check_full_env.sh` - Verifies GPU, CUDA, Python, dependencies

- **`run_plot_latency.sh`**: Generate latency comparison plots
  - `bash scripts/run_plot_latency.sh [output_dir]` - Default: `results/fig`

- **`run_multigpu_test.sh`**: Test multi-GPU memory usage
  - `bash scripts/run_multigpu_test.sh` - Memory profiling for long sequences

#### Evaluation Scripts (Optional)

- **`run_longbench.sh`**: Run LongBench evaluation
  - `bash scripts/run_longbench.sh [task] [type] [num_samples]` - Default: hotpotqa, baseline, 30

- **`run_longbench_setup.sh`**: Setup LongBench evaluation
  - `bash scripts/run_longbench_setup.sh [task] [model] [pruning_module] [output]`

- **`run_ablation.sh`**: Run ablation study
  - `bash scripts/run_ablation.sh` - Generates ablation results

- **`run_cooperative_replanning.sh`**: Cooperative multi-agent planning with E-RECAP
  - `bash scripts/run_cooperative_replanning.sh --keep_ratio 0.7 --save_results` - Run with default task
  - `bash scripts/run_cooperative_replanning.sh --task_type embodied` - Run embodied replanning scenario

## Key Features

- **Cost-Aware Pruning**: Remove redundant tokens during prefill to reduce computation (up to 71% token reduction, 2-40× speedup depending on sequence length and GPU configuration)
- **Layer-wise Pruning**: Progressive pruning across Transformer layers (8 pruning points: layers 4, 7, 10, 13, 16, 19, 22, 25)
- **Multi-GPU Support**: Automatic distributed inference for long sequences (tested up to 32K tokens, achieving 20.7× average speedup on 8× RTX 5880)
- **Learnable Pruning Module**: Lightweight MLP (hidden_size → hidden_size/4 → 1) trained on instruction-following data
- **Cooperative Multi-Agent Planning**: Sequential multi-agent replanning with E-RECAP context pruning (K=2-8 agents, see [Multi-Agent Planning](#multi-agent-planning))
- **Quality Preservation**: Maintains task success rate while significantly reducing computation (typically <2% quality degradation at keep_ratio=0.7)

<!--
## Results

**Note:** The following results are for Qwen2-7B model. Prefill results are for planning task phase only. End-to-end results include both prefill and decode phases.

### Single GPU Prefill Speedup

| Keep Ratio | Average Speedup | Range |
|------------|----------------|-------|
| 0.9 | 1.43× | 1.38-1.50× |
| 0.8 | 1.96× | 1.88-2.09× |
| 0.7 | 2.48× | 2.35-2.66× |

**Hardware:** NVIDIA RTX 5880 Ada (48GB), single GPU

### Multi-GPU Prefill Speedup

| Sequence Length | Speedup | Latency Reduction |
|----------------|---------|-------------------|
| 1024 | 12.45× | 92.0% |
| 2048 | 13.12× | 92.4% |
| 4096 | 14.84× | 93.3% |
| 8192 | 17.23× | 94.2% |
| 16384 | 26.73× | 96.3% |
| 32768 | 39.69× | 97.5% |

**Average:** 20.68× speedup, 94.3% latency reduction

**Hardware:** 8× NVIDIA RTX 5880 Ada Generation (48GB each), keep_ratio=0.7
-->

<!--
### Single GPU End-to-End Speedup

| Keep Ratio | Sequence Length | Speedup |
|------------|----------------|---------|
| 0.7 | 1024 | 1.53× |
| 0.7 | 2048 | 5.50× |
| 0.7 | 4096 | 0.66× |
| 0.7 | 8192 | 5.15× |
| 0.7 | 16384 | 4.86× |
| 0.7 | 32768 | 2.54× |

**Hardware:** NVIDIA RTX 5880 Ada (48GB), single GPU
-->

## Multi-Agent Planning

E-RECAP supports cooperative multi-agent planning where multiple agents operate sequentially, each receiving a shared planning context pruned by E-RECAP's cost-aware token pruning module. This setting captures multi-agent replanning characteristics—context growth, information aggregation, and iterative plan revision—while maintaining strict control over experimental variables.

**Note:** This is a planning-level multi-agent setting (not multi-robot physical control). Multiple planning agents contribute information sequentially to a shared context, which is pruned by E-RECAP before each agent invocation. This design systematically amplifies context growth to evaluate E-RECAP's scalability (K=2-8 agents).

**Quick example (uses default task, no input required):**
```bash
# Run with default task description
bash scripts/run_cooperative_replanning.sh --keep_ratio 0.7 --save_results

# Or Python (default task included)
python3 src/multi_agent/run_cooperative_test.py --keep_ratio 0.7 --save_results
```

**Key features:**
- **Shared Context Buffer**: Accumulates task descriptions, plans, constraints, and agent contributions
- **E-RECAP Pruning**: Context pruned before each agent invocation to control growth
- **Structured Output**: Agent outputs in structured format (observations, conflicts, plan patches)
- **Framework Compatible**: Optional CrewAI/LangChain support (see below)

**Optional Framework Support (CrewAI/LangChain):**

To enable CrewAI or LangChain integration, install the optional dependencies:
```bash
pip install crewai>=0.28.8 langchain>=0.1.17 langchain-community>=0.1.17
```

Then copy the agent configuration file to the framework_optional directory (if you have one):
```bash
# Create framework_optional directory if it doesn't exist
mkdir -p src/multi_agent/framework_optional

# Copy your agents_config.json file to the framework_optional directory
# The file should follow the format expected by CrewAI/LangChain
```

Note: Framework support is optional. The default implementation works without CrewAI/LangChain. When enabled, frameworks are used only for scheduling/role-assignment, while prompt construction, context buffering, and pruning remain under E-RECAP control. See `src/multi_agent/framework_wrapper.py` for implementation details.

**Baseline comparison:**
```bash
# Run baseline (no pruning) for comparison
bash scripts/run_cooperative_replanning.sh --baseline --save_results

# Compare baseline vs E-RECAP results
python3 src/multi_agent/compare_baseline_erecap.py \
    --baseline_file results/cooperative_planning_cooperative_baseline.json \
    --erecap_file results/cooperative_planning_cooperative_0.7.json
```

For detailed implementation, see `paper/part3_sum.md`.

## Embodied Evaluation (Planned)

E-RECAP is designed for embodied AI replanning scenarios. Planned evaluation includes:

- **Platform**: [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) ([PointNav](https://aihabitat.org/docs/habitat-lab/habitat_task.html#pointnav), [ObjectNav](https://aihabitat.org/docs/habitat-lab/habitat_task.html#objectnav) tasks)
- **Scenes**: [Matterport3D (MP3D)](https://niessner.github.io/Matterport/), [Gibson](http://gibsonenv.stanford.edu/), [Replica](https://github.com/facebookresearch/Replica-Dataset)
- **Setting**: Cooperative multi-agent replanning (K=2-8 agents)
- **Metrics**: Success Rate, SPL, token cost, latency, replanning frequency
- **Baselines**: No-Pruning, Random-Pruning, Heuristic-Pruning

The Habitat integration will evaluate E-RECAP's effectiveness in real embodied replanning scenarios where context naturally grows through plan-execute-observe-replan cycles. See `paper/habitat_integration_design.md` for detailed design.

## Model Configuration

E-RECAP supports any HuggingFace-compatible Transformer model. To use a different model:

1. **Place model files** in `checkpoints/<your-model-name>/`

2. **Update model path** in the following files:
   - `src/inference_erecap.py`: Set `MODEL_PATH = "checkpoints/<your-model-name>"`
   - `src/stage1_saliency.py`: Set `MODEL_PATH` (if running Stage 1)
   - `src/stage2_pruning.py`: Set `MODEL_NAME = "checkpoints/<your-model-name>"` (if running Stage 2)

3. **Train pruning module** (if switching to a model with different `hidden_size`):
   - The pruning module is model-specific and depends on the model's `hidden_size`
   - If your new model has the same `hidden_size`, you can reuse the existing `pruning_module.pt`
   - Otherwise, retrain by running Stage 2 with the new model

