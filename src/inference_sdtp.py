import argparse
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "checkpoints/qwen2-7b-instruct"
PRUNING_CKPT = "checkpoints/pruning_module.pt"
MAX_NEW_TOKENS = 128
PRUNE_LAYERS = [4, 7, 10, 13, 16, 19, 22, 25, 28, 31]
KEEP_RATIO = 0.7
MIN_HEAD_TOKENS = 4
MIN_TAIL_TOKENS = 16


class TokenPruningModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states):
        logits = self.scorer(hidden_states).squeeze(-1)
        return logits


def load_model_and_pruners():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=None,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    hidden_size = model.config.hidden_size

    pruning_modules = nn.ModuleDict()
    for idx in PRUNE_LAYERS:
        pruning_modules[str(idx)] = TokenPruningModule(hidden_size)

    state_dict = torch.load(PRUNING_CKPT, map_location="cpu")
    pruning_modules.load_state_dict(state_dict)
    pruning_modules.to(device)
    pruning_modules.eval()
    for p in pruning_modules.parameters():
        p.requires_grad = False

    return model, tokenizer, pruning_modules


def apply_token_pruning(hidden_states, pruning_module, keep_ratio=KEEP_RATIO):
    seq_len = hidden_states.size(1)
    device_local = hidden_states.device
    hs_flat = hidden_states.squeeze(0)
    scores = pruning_module(hs_flat)

    base_keep = set(range(min(MIN_HEAD_TOKENS, seq_len)))
    for i in range(max(seq_len - MIN_TAIL_TOKENS, 0), seq_len):
        base_keep.add(i)

    target_keep = max(int(seq_len * keep_ratio), len(base_keep))
    target_keep = min(target_keep, seq_len)

    _, top_indices = torch.topk(scores, k=seq_len)
    selected = []
    for idx in top_indices.tolist():
        if idx in base_keep:
            selected.append(idx)
    for idx in top_indices.tolist():
        if idx not in base_keep and len(selected) < target_keep:
            selected.append(idx)

    selected = sorted(set(selected))
    index_tensor = torch.tensor(selected, device=device_local, dtype=torch.long)
    pruned_hs = hidden_states[:, index_tensor, :]

    return pruned_hs, index_tensor


def prefill_with_pruning(model, input_ids, attention_mask, pruning_modules, keep_ratio=KEEP_RATIO):
    model_inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    hidden_states = model.model.embed_tokens(model_inputs["input_ids"])
    attn_mask = model_inputs["attention_mask"]

    for layer_idx, block in enumerate(model.model.layers):
        block_outputs = block(hidden_states, attention_mask=attn_mask)
        hidden_states = block_outputs[0]

        if layer_idx in PRUNE_LAYERS:
            module = pruning_modules[str(layer_idx)]
            hidden_states, kept_idx = apply_token_pruning(hidden_states, module, keep_ratio)
            attn_mask = attn_mask[:, kept_idx]

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits


def baseline_prefill(model, input_ids, attention_mask):
    model_inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    with torch.no_grad():
        outputs = model(
            **model_inputs,
            use_cache=False,
        )
    return outputs.logits


def measure_latency(fn, *args, warmup=2, runs=5):
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


def build_dummy_input(tokenizer, length):
    prompt = "Hello, this is a test."
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if ids.size(0) >= length:
        input_ids = ids[:length].unsqueeze(0)
    else:
        repeat_times = (length + ids.size(0) - 1) // ids.size(0)
        tiled = ids.repeat(repeat_times)[:length]
        input_ids = tiled.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    return input_ids.to(device), attention_mask.to(device)


def profile_lengths(lengths, keep_ratio):
    model, tokenizer, pruning_modules = load_model_and_pruners()
    model.eval()

    print("Profiling lengths:", lengths)
    for length in lengths:
        input_ids, attention_mask = build_dummy_input(tokenizer, length)

        base_time = measure_latency(
            lambda ids, mask: baseline_prefill(model, ids, mask),
            input_ids,
            attention_mask,
        )
        sdtp_time = measure_latency(
            lambda ids, mask: prefill_with_pruning(model, ids, mask, pruning_modules, keep_ratio),
            input_ids,
            attention_mask,
        )
        speedup = base_time / sdtp_time if sdtp_time > 0 else float("inf")
        print(f"Length {length}: baseline={base_time:.4f}s, sdtp={sdtp_time:.4f}s, speedup={speedup:.2f}x")


def generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="profile",
        choices=["profile", "generate"],
    )
    parser.add_argument("--prompt", type=str, default="Hello, SDTP!")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--keep_ratio", type=float, default=KEEP_RATIO)
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
