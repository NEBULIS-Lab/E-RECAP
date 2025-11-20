# SDTP 推理（单机单卡的完整推理脚本）

import argparse
import time
import json
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Local model and pruning ckpt
# ============================
MODEL_PATH = "checkpoints/qwen2-7b-instruct"
PRUNING_CKPT = "checkpoints/pruning_module.pt"

# ============================
# Pruning config (must match Stage2)
# ============================
MAX_NEW_TOKENS = 128

# Configuration presets
KEEP09_CONFIG = {
    "prune_layers": [4, 7, 10, 13, 16, 19, 22, 25],  # 8 layers
    "keep_ratio": 0.9,  # Keep 90% tokens per layer
    "min_head_tokens": 4,
    "min_tail_ratio": 0.1,  # Keep 10% of tokens at tail (as in paper)
    "cumulative_keep_ratio": 0.9 ** 8,  # ~0.43
}

KEEP08_CONFIG = {
    "prune_layers": [4, 7, 10, 13, 16, 19, 22, 25],  # 8 layers
    "keep_ratio": 0.8,  # Keep 80% tokens per layer
    "min_head_tokens": 4,
    "min_tail_ratio": 0.1,  # Keep 10% of tokens at tail (as in paper)
    "cumulative_keep_ratio": 0.8 ** 8,  # ~0.17
}

KEEP07_CONFIG = {
    "prune_layers": [4, 7, 10, 13, 16, 19, 22, 25],  # 8 layers
    "keep_ratio": 0.7,  # Keep 70% tokens per layer
    "min_head_tokens": 4,
    "min_tail_ratio": 0.1,  # Keep 10% of tokens at tail (as in paper)
    "cumulative_keep_ratio": 0.7 ** 8,  # ~0.058
}

# Default config (keep07)
PRUNE_LAYERS = KEEP07_CONFIG["prune_layers"]
KEEP_RATIO = KEEP07_CONFIG["keep_ratio"]
MIN_HEAD_TOKENS = KEEP07_CONFIG["min_head_tokens"]
MIN_TAIL_RATIO = KEEP07_CONFIG["min_tail_ratio"]


# ============================
# TokenPruningModule
# ============================
class TokenPruningModule(nn.Module):
    """Small MLP that outputs an importance score per token."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),  # Use GELU as in paper (was ReLU)
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [seq_len, hidden] or [batch, seq_len, hidden]
        returns: [seq_len]
        """
        return self.scorer(hidden_states).squeeze(-1)


# ============================
# Load model + pruning modules
# ============================
def load_model_and_pruners(prune_layers=None):
    """
    Load model and pruning modules.
    
    Args:
        prune_layers: List of layer indices to prune. If None, uses PRUNE_LAYERS.
    
    Returns:
        model, tokenizer, pruning_modules
    """
    if prune_layers is None:
        prune_layers = PRUNE_LAYERS
    
    # Load Qwen2 model in float16 on GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=None,
        local_files_only=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    hidden_size = model.config.hidden_size

    # Build pruning modules for selected layers
    pruning_modules = nn.ModuleDict(
        {str(i): TokenPruningModule(hidden_size) for i in prune_layers}
    )

    # Load trained pruning weights from Stage2
    state_dict = torch.load(PRUNING_CKPT, map_location="cpu")
    pruning_modules.load_state_dict(state_dict)

    pruning_modules.to(device)
    pruning_modules.half()
    pruning_modules.eval()
    for p in pruning_modules.parameters():
        p.requires_grad = False

    return model, tokenizer, pruning_modules


# ============================
# Pruning logic
# ============================
def apply_token_pruning(
    hidden_states: torch.Tensor,
    pruning_module: nn.Module,
    keep_ratio: float,
    min_head_tokens: int = None,
    min_tail_ratio: float = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    hidden_states: [1, seq_len, hidden]
    pruning_module: TokenPruningModule
    keep_ratio: fraction of tokens to keep
    min_head_tokens: minimum tokens to keep at head (default: MIN_HEAD_TOKENS)
    min_tail_ratio: ratio of tokens to keep at tail (default: MIN_TAIL_RATIO, as in paper: 10%)

    Returns:
        pruned_hidden_states: [1, kept_len, hidden]
        index_tensor: [kept_len] indices kept
        stats: dict with pruning statistics
    """
    if min_head_tokens is None:
        min_head_tokens = MIN_HEAD_TOKENS
    if min_tail_ratio is None:
        min_tail_ratio = MIN_TAIL_RATIO
    
    seq_len = hidden_states.size(1)

    # [seq_len, hidden]
    hs_flat = hidden_states.squeeze(0)

    # importance scores: [seq_len]
    scores = pruning_module(hs_flat)

    # Always keep the first min_head_tokens and last min_tail_ratio * seq_len tokens (as in paper)
    base_keep = set(range(min(min_head_tokens, seq_len)))
    min_tail_tokens = max(16, int(seq_len * min_tail_ratio))  # At least 16, or 10% of sequence
    for i in range(max(0, seq_len - min_tail_tokens), seq_len):
        base_keep.add(i)

    # How many tokens to keep in total
    target_keep = max(int(seq_len * keep_ratio), len(base_keep))
    target_keep = min(target_keep, seq_len)

    # Sort tokens by score (descending)
    _, sorted_idx = torch.topk(scores, k=seq_len)

    # First add mandatory tokens, then fill up with highest scores
    selected = []
    for idx in sorted_idx.tolist():
        if idx in base_keep:
            selected.append(idx)
    for idx in sorted_idx.tolist():
        if idx not in base_keep and len(selected) < target_keep:
            selected.append(idx)

    selected = sorted(selected)
    index_tensor = torch.tensor(
        selected,
        device=hidden_states.device,
        dtype=torch.long,
    )
    pruned_hidden = hidden_states[:, index_tensor, :]

    # Calculate statistics
    tokens_kept = len(selected)
    tokens_pruned = seq_len - tokens_kept
    pruning_ratio = tokens_pruned / seq_len if seq_len > 0 else 0.0

    stats = {
        "tokens_kept": tokens_kept,
        "tokens_pruned": tokens_pruned,
        "pruning_ratio": pruning_ratio,
        "original_length": seq_len,
    }

    return pruned_hidden, index_tensor, stats


# ============================
# Prefill with pruning (SDTP)
# ============================
def prefill_with_pruning(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,  # not used internally, kept for API symmetry
    pruning_modules: nn.ModuleDict,
    keep_ratio: float,
    prune_layers: list = None,
    min_head_tokens: int = None,
    min_tail_ratio: float = None,
) -> tuple[torch.Tensor, dict]:
    """
    Manual forward over Transformer layers with token pruning applied
    at selected layers. Mirrors Stage2 training:

    - Start from embed_tokens(input_ids).
    - For each layer:
        * Build position_ids explicitly.
        * Call layer(hidden_states, position_ids=..., attention_mask=None).
        * Optionally prune tokens at this layer.
    - Apply final norm + lm_head to get logits.
    
    Returns:
        logits: Model output logits
        pruning_stats: Dict with aggregated pruning statistics
    """
    if prune_layers is None:
        prune_layers = PRUNE_LAYERS
    if min_head_tokens is None:
        min_head_tokens = MIN_HEAD_TOKENS
    if min_tail_ratio is None:
        min_tail_ratio = MIN_TAIL_RATIO

    # Embed tokens: [1, seq_len, hidden]
    hidden_states = model.model.embed_tokens(input_ids)
    original_length = hidden_states.size(1)
    
    # Track pruning statistics
    pruning_stats = {
        "total_pruning_steps": 0,
        "total_tokens_pruned": 0,
        "final_length": original_length,
        "layer_stats": []
    }

    for layer_idx, layer in enumerate(model.model.layers):
        # Build position ids: [1, seq_len]
        position_ids = torch.arange(
            0,
            hidden_states.size(1),
            dtype=torch.long,
            device=hidden_states.device,
        ).unsqueeze(0)

        # Use internal causal mask: attention_mask=None
        outputs = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = outputs[0]

        # Apply token pruning on selected layers
        if layer_idx in prune_layers:
            pruner = pruning_modules[str(layer_idx)]
            hidden_states, _, stats = apply_token_pruning(
                hidden_states,
                pruner,
                keep_ratio,
                min_head_tokens,
                min_tail_ratio,
            )
            pruning_stats["total_pruning_steps"] += 1
            pruning_stats["total_tokens_pruned"] += stats["tokens_pruned"]
            pruning_stats["final_length"] = stats["tokens_kept"]
            pruning_stats["layer_stats"].append({
                "layer": layer_idx,
                **stats
            })

    # Final RMSNorm + LM head to get logits
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits, pruning_stats


# ============================
# Baseline prefill (no pruning)
# ============================
def baseline_prefill(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Baseline reference: normal model forward using
    prepare_inputs_for_generation and full sequence.
    """
    model_inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    with torch.no_grad():
        outputs = model(**model_inputs)
    return outputs.logits


# ============================
# Timing utilities
# ============================
def measure_latency(fn, *args, warmup: int = 1, runs: int = 3) -> float:
    """
    Measure average latency of fn(*args).
    """
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = fn(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


def build_dummy_input(tokenizer: AutoTokenizer, length: int):
    """
    Build a fake long sequence of given length by repeating a short prompt.
    """
    base_ids = tokenizer("Hello, this is a test.", return_tensors="pt")[
        "input_ids"
    ][0]
    if base_ids.size(0) >= length:
        ids = base_ids[:length]
    else:
        repeat = (length + base_ids.size(0) - 1) // base_ids.size(0)
        ids = base_ids.repeat(repeat)[:length]

    input_ids = ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return input_ids, attention_mask


# ============================
# Profiling
# ============================
def get_hardware_info():
    """Collect hardware information."""
    info = {
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["gpu_model"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["cuda_version"] = torch.version.cuda
    
    return info


def profile_lengths(lengths, config_name: str = "keep07", save_json: bool = True):
    """
    Profile baseline vs SDTP for given sequence lengths.
    
    Args:
        lengths: List of sequence lengths to test
        config_name: Configuration name ("keep09", "keep08", or "keep07")
        save_json: Whether to save results to JSON
    """
    # Select configuration
    if config_name == "keep09":
        config = KEEP09_CONFIG
    elif config_name == "keep08":
        config = KEEP08_CONFIG
    else:
        config = KEEP07_CONFIG
    
    print(f"[Config] Using {config_name} configuration:")
    print(f"  Prune layers: {config['prune_layers']}")
    print(f"  Keep ratio: {config['keep_ratio']}")
    print(f"  Cumulative keep ratio: {config['cumulative_keep_ratio']:.4f}")
    
    # Load model with selected configuration
    model, tokenizer, pruners = load_model_and_pruners(prune_layers=config["prune_layers"])
    model.eval()

    print("Profiling lengths:", lengths)
    
    # Store results in new format
    results_data = {
        "metadata": {
            "config_name": config_name,
            "model": os.path.basename(MODEL_PATH),
            "hardware": get_hardware_info(),
            "pruning_config": {
                "prune_layers": config["prune_layers"],
                "keep_ratio": config["keep_ratio"],
                "min_head_tokens": config["min_head_tokens"],
                "min_tail_ratio": config["min_tail_ratio"],
                "cumulative_keep_ratio": config["cumulative_keep_ratio"],
            },
            "run_info": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "script": "inference_sdtp.py",
                "mode": "profile",
            }
        },
        "results": {},
        "summary": {
            "tested_lengths": [],
            "avg_speedup": 0.0,
            "avg_memory_reduction": 0.0,
        }
    }
    
    speedups = []
    memory_reductions = []
    
    for L in lengths:
        # Build new input for each length
        input_ids, attention_mask = build_dummy_input(tokenizer, L)

        try:
            # Reset memory tracking
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            
            # Baseline: no pruning
            baseline_t = measure_latency(
                lambda x, m: baseline_prefill(model, x, m),
                input_ids,
                attention_mask,
            )
            
            baseline_memory = 0.0
            if device.type == "cuda":
                baseline_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                torch.cuda.reset_peak_memory_stats()

            # SDTP: manual forward + pruning
            def sdtp_fn(x, m):
                logits, stats = prefill_with_pruning(
                    model, x, m, pruners, 
                    config["keep_ratio"],
                    config["prune_layers"],
                    config["min_head_tokens"],
                    config["min_tail_ratio"],
                )
                return logits, stats
            
            sdtp_t = measure_latency(
                lambda x, m: sdtp_fn(x, m)[0],  # Only measure latency, not stats
                input_ids,
                attention_mask,
            )
            
            # Get pruning stats from a separate run
            _, pruning_stats = sdtp_fn(input_ids, attention_mask)
            
            sdtp_memory = 0.0
            if device.type == "cuda":
                sdtp_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB

            speedup = baseline_t / sdtp_t if sdtp_t > 0 else float("inf")
            memory_reduction = (baseline_memory - sdtp_memory) / baseline_memory if baseline_memory > 0 else 0.0
            
            print(
                f"[Length {L}] baseline={baseline_t:.4f}s ({baseline_memory:.2f}GB)  "
                f"sdtp={sdtp_t:.4f}s ({sdtp_memory:.2f}GB)  "
                f"speedup={speedup:.2f}x  memory_reduction={memory_reduction:.2%}"
            )
            
            # Store detailed results
            results_data["results"][str(L)] = {
                "baseline": {
                    "latency_seconds": baseline_t,
                    "memory_gb": baseline_memory,
                },
                "sdtp": {
                    "latency_seconds": sdtp_t,
                    "memory_gb": sdtp_memory,
                    "tokens_kept": pruning_stats["final_length"],
                    "tokens_pruned": pruning_stats["total_tokens_pruned"],
                    "pruning_ratio": pruning_stats["total_tokens_pruned"] / L if L > 0 else 0.0,
                    "pruning_steps": pruning_stats["total_pruning_steps"],
                },
                "speedup": speedup,
                "memory_reduction": memory_reduction,
            }
            
            speedups.append(speedup)
            memory_reductions.append(memory_reduction)
            results_data["summary"]["tested_lengths"].append(L)

        except torch.cuda.OutOfMemoryError:
            print(f"[Length {L}] OOM on GPU, skipping this length.")
        except Exception as e:
            print(f"[Length {L}] Error: {e}")
        finally:
            # Explicitly free tensors and clear cache to avoid accumulation
            if 'input_ids' in locals():
                del input_ids, attention_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
    
    # Calculate summary statistics
    if speedups:
        results_data["summary"]["avg_speedup"] = sum(speedups) / len(speedups)
    if memory_reductions:
        results_data["summary"]["avg_memory_reduction"] = sum(memory_reductions) / len(memory_reductions)
    
    # Save results to JSON
    if save_json and results_data["results"]:
        os.makedirs("results", exist_ok=True)
        
        # Save combined results (new format)
        combined_path = f"results/latency_results_{config_name}.json"
        with open(combined_path, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"[OK] Results saved to {combined_path}")
        
        # Also save backward-compatible separate files
        baseline_results = {k: v["baseline"]["latency_seconds"] 
                           for k, v in results_data["results"].items()}
        sdtp_results = {k: v["sdtp"]["latency_seconds"] 
                       for k, v in results_data["results"].items()}
        
        baseline_path = f"results/latency_baseline_{config_name}.json"
        sdtp_path = f"results/latency_sdtp_{config_name}.json"
        
        with open(baseline_path, "w") as f:
            json.dump(baseline_results, f, indent=2)
        print(f"[OK] Baseline results saved to {baseline_path}")
        
        with open(sdtp_path, "w") as f:
            json.dump(sdtp_results, f, indent=2)
        print(f"[OK] SDTP results saved to {sdtp_path}")


# ============================
# Text generation (baseline)
# ============================
def generate_text(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


# ============================
# CLI
# ============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="profile",
        choices=["profile", "generate"],
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, SDTP!",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=None,
        help="Token keep ratio for pruning (overrides config if set)",
    )
    parser.add_argument(
        "--config",
        choices=["keep09", "keep08", "keep07"],
        default="keep07",
        help="Configuration preset: keep09 (0.9), keep08 (0.8), or keep07 (0.7) keep ratio",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "profile":
        profile_lengths(args.lengths, config_name=args.config)

    elif args.mode == "generate":
        model, tokenizer, _ = load_model_and_pruners()
        model.eval()
        text = generate_text(model, tokenizer, args.prompt)
        print(text)


if __name__ == "__main__":
    main()
