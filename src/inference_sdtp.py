import argparse
import time

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
PRUNE_LAYERS = [4, 7, 10, 13, 16, 19, 22, 25]
KEEP_RATIO = 0.7
MIN_HEAD_TOKENS = 4
MIN_TAIL_TOKENS = 16


# ============================
# TokenPruningModule
# ============================
class TokenPruningModule(nn.Module):
    """Small MLP that outputs an importance score per token."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
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
def load_model_and_pruners():
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
        {str(i): TokenPruningModule(hidden_size) for i in PRUNE_LAYERS}
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    hidden_states: [1, seq_len, hidden]
    pruning_module: TokenPruningModule
    keep_ratio: fraction of tokens to keep

    Returns:
        pruned_hidden_states: [1, kept_len, hidden]
        index_tensor: [kept_len] indices kept
    """
    seq_len = hidden_states.size(1)

    # [seq_len, hidden]
    hs_flat = hidden_states.squeeze(0)

    # importance scores: [seq_len]
    scores = pruning_module(hs_flat)

    # Always keep the first MIN_HEAD_TOKENS and last MIN_TAIL_TOKENS
    base_keep = set(range(min(MIN_HEAD_TOKENS, seq_len)))
    for i in range(max(0, seq_len - MIN_TAIL_TOKENS), seq_len):
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

    return pruned_hidden, index_tensor


# ============================
# Prefill with pruning (SDTP)
# ============================
def prefill_with_pruning(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,  # not used internally, kept for API symmetry
    pruning_modules: nn.ModuleDict,
    keep_ratio: float,
) -> torch.Tensor:
    """
    Manual forward over Transformer layers with token pruning applied
    at selected layers. Mirrors Stage2 training:

    - Start from embed_tokens(input_ids).
    - For each layer:
        * Build position_ids explicitly.
        * Call layer(hidden_states, position_ids=..., attention_mask=None).
        * Optionally prune tokens at this layer.
    - Apply final norm + lm_head to get logits.
    """

    # Embed tokens: [1, seq_len, hidden]
    hidden_states = model.model.embed_tokens(input_ids)

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
        if layer_idx in PRUNE_LAYERS:
            pruner = pruning_modules[str(layer_idx)]
            hidden_states, _ = apply_token_pruning(
                hidden_states,
                pruner,
                keep_ratio,
            )

    # Final RMSNorm + LM head to get logits
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits


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
def profile_lengths(lengths, keep_ratio: float):
    model, tokenizer, pruners = load_model_and_pruners()
    model.eval()

    print("Profiling lengths:", lengths)
    for L in lengths:
        # Build new input for each length
        input_ids, attention_mask = build_dummy_input(tokenizer, L)

        try:
            # Baseline: no pruning
            baseline_t = measure_latency(
                lambda x, m: baseline_prefill(model, x, m),
                input_ids,
                attention_mask,
            )

            # SDTP: manual forward + pruning
            sdtp_t = measure_latency(
                lambda x, m: prefill_with_pruning(
                    model, x, m, pruners, keep_ratio
                ),
                input_ids,
                attention_mask,
            )

            speedup = baseline_t / sdtp_t if sdtp_t > 0 else float("inf")
            print(
                f"[Length {L}] baseline={baseline_t:.4f}s  "
                f"sdtp={sdtp_t:.4f}s  speedup={speedup:.2f}x"
            )

        except torch.cuda.OutOfMemoryError:
            print(f"[Length {L}] OOM on GPU, skipping this length.")
        finally:
            # Explicitly free tensors and clear cache to avoid accumulation
            del input_ids, attention_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()


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
        default=[4096, 8192, 16384, 32768],
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=KEEP_RATIO,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "profile":
        profile_lengths(args.lengths, args.keep_ratio)

    elif args.mode == "generate":
        model, tokenizer, _ = load_model_and_pruners()
        model.eval()
        text = generate_text(model, tokenizer, args.prompt)
        print(text)


if __name__ == "__main__":
    main()
