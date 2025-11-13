import argparse
import time
import torch
import torch.nn as nn
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================
# Config
# ============================
MODEL_PATH = "checkpoints/qwen2-7b-instruct"
PRUNING_CKPT = "checkpoints/pruning_module.pt"

PRUNE_LAYERS = [4, 7, 10, 13, 16, 19, 22, 25]
KEEP_RATIO = 0.7
MIN_HEAD_TOKENS = 4
MIN_TAIL_TOKENS = 16
MAX_NEW_TOKENS = 128


# ============================
# Multi-GPU helper
# ============================
def print_gpu_usage(tag=""):
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    print(f"[{tag}] GPU0: used={info.used//1024**2}MB total={info.total//1024**2}MB")


# ============================
# Pruning MLP
# ============================
class TokenPruningModule(nn.Module):
    """Small token-scoring MLP"""
    def __init__(self, hidden_size):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states):
        return self.scorer(hidden_states).squeeze(-1)


# ============================
# Load model + pruners
# ============================
def load_model_and_pruners():
    print("[Loading model] device_map=auto ...")

    # Qwen2 model automatically split across GPUs
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,     # ensures half precision across devices
        device_map="auto",
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    hidden_size = model.config.hidden_size

    # Build pruning modules
    pruning_modules = nn.ModuleDict({
        str(i): TokenPruningModule(hidden_size)
        for i in PRUNE_LAYERS
    })

    # Load trained weights
    state_dict = torch.load(PRUNING_CKPT, map_location="cpu")
    pruning_modules.load_state_dict(state_dict)

    # *** KEY FIX: convert pruning modules to FP16 ***
    pruning_modules.half()
    pruning_modules.eval()
    for p in pruning_modules.parameters():
        p.requires_grad = False

    print("[Model + pruners loaded]")
    return model, tokenizer, pruning_modules


# ============================
# Token pruning logic
# ============================
def apply_token_pruning(hidden_states, pruning_module, keep_ratio):
    """
    hidden_states: [1, seq, hidden]
    """
    device = hidden_states.device
    dtype = hidden_states.dtype

    # ensure pruning module matches dtype
    pruning_module = pruning_module.to(dtype=dtype, device=device)

    seq_len = hidden_states.size(1)

    # flatten to [seq, hidden]
    hs_flat = hidden_states.squeeze(0)   # [seq, hidden]

    # scores: [seq]
    scores = pruning_module(hs_flat)

    # mandatory keep: head + tail
    base_keep = set(range(min(MIN_HEAD_TOKENS, seq_len)))
    for i in range(max(0, seq_len - MIN_TAIL_TOKENS), seq_len):
        base_keep.add(i)

    target_keep = max(int(seq_len * keep_ratio), len(base_keep))
    target_keep = min(target_keep, seq_len)

    # sort by score
    _, sorted_idx = torch.topk(scores, k=seq_len)

    selected = []
    for idx in sorted_idx.tolist():
        if idx in base_keep:
            selected.append(idx)
    for idx in sorted_idx.tolist():
        if idx not in base_keep and len(selected) < target_keep:
            selected.append(idx)

    selected = sorted(selected)
    index_tensor = torch.tensor(selected, dtype=torch.long, device=device)

    pruned_hidden = hidden_states[:, index_tensor, :]

    return pruned_hidden, index_tensor


# ============================
# Manual SDTP forward
# ============================
def prefill_with_pruning(model, input_ids, attention_mask, pruners, keep_ratio):
    """
    Manual forward with pruning at selected layers.
    Supports multi-GPU (device_map=auto)
    """

    # embed tokens will be on the correct GPU automatically
    hidden_states = model.model.embed_tokens(input_ids)

    for layer_idx, layer in enumerate(model.model.layers):

        device = hidden_states.device
        seq_len = hidden_states.size(1)

        # build position ids
        position_ids = torch.arange(
            0, seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)

        # transformer block forward
        outputs = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = outputs[0]

        # pruning
        if layer_idx in PRUNE_LAYERS:
            pruner = pruners[str(layer_idx)]
            hidden_states, _ = apply_token_pruning(
                hidden_states,
                pruner,
                keep_ratio,
            )

    # final norm + head
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits


# ============================
# Baseline (normal forward)
# ============================
def baseline_prefill(model, input_ids, attention_mask):
    inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits


# ============================
# Timing utility
# ============================
def measure_latency(fn, *args, warmup=1, runs=3):
    for _ in range(warmup):
        _ = fn(*args)
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)


# ============================
# Dummy input
# ============================
def build_dummy_input(tokenizer, length, device="cuda"):
    base_ids = tokenizer("Hello, this is a test.", return_tensors="pt")["input_ids"][0]
    if base_ids.size(0) >= length:
        ids = base_ids[:length]
    else:
        repeat = (length + base_ids.size(0) - 1) // base_ids.size(0)
        ids = base_ids.repeat(repeat)[:length]

    input_ids = ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return input_ids, attention_mask


# ============================
# Profiling multi-GPU
# ============================
def profile_lengths(lengths, keep_ratio):
    model, tokenizer, pruners = load_model_and_pruners()
    model.eval()

    print("Profiling lengths:", lengths)

    for L in lengths:
        try:
            print("\n======================================")
            print(f"Testing length {L}")
            print("======================================")

            # Find first device holding first layer (model.model.layers[0])
            first_device = next(model.model.layers[0].parameters()).device

            input_ids, attention_mask = build_dummy_input(tokenizer, L, device=first_device)

            baseline_t = measure_latency(
                lambda x, m: baseline_prefill(model, x, m),
                input_ids, attention_mask
            )
            sdtp_t = measure_latency(
                lambda x, m: prefill_with_pruning(model, x, m, pruners, keep_ratio),
                input_ids, attention_mask
            )

            print(f"[Length {L}] baseline={baseline_t:.4f}s  sdtp={sdtp_t:.4f}s  speedup={baseline_t/sdtp_t:.2f}x")

        except torch.cuda.OutOfMemoryError:
            print(f"[Length {L}] OOM, skipped.")
        finally:
            del input_ids, attention_mask
            torch.cuda.empty_cache()


# ============================
# Text generation (multi-GPU)
# ============================
def generate_text(prompt):
    model, tokenizer, pruners = load_model_and_pruners()
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ============================
# Entry
# ============================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["profile", "generate"], default="profile")
    p.add_argument("--prompt", type=str, default="Hello SDTP multi GPU!")
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    p.add_argument("--keep_ratio", type=float, default=KEEP_RATIO)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "profile":
        profile_lengths(args.lengths, args.keep_ratio)
    else:
        print(generate_text(args.prompt))


if __name__ == "__main__":
    main()

