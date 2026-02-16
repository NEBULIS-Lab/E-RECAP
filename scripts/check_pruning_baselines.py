#!/usr/bin/env python3
"""Sanity-check paper baselines: Random-Pruning / Heuristic-Pruning(recency) / E-RECAP.

This script validates that under the same progressive pruning schedule
(same prune_layers, keep_ratio, head/tail preservation), different token
selection strategies:
  - keep identical token counts (per layer and final)
  - are deterministic for Random-Pruning under a fixed seed

It is intended as a fast correctness check before running Habitat episodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys as _sys  # noqa: E402

_sys.path.insert(0, str(PROJECT_ROOT / "src"))
from inference_erecap import prefill_with_pruning, TokenPruningModule  # noqa: E402


def _build_ids(tokenizer, dolly_jsonl: Path, target_len: int, seed: int) -> torch.Tensor:
    import random

    rng = random.Random(int(seed))
    lines = dolly_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise RuntimeError(f"Empty dolly jsonl: {dolly_jsonl}")
    ids_chunks: list[torch.Tensor] = []
    total = 0
    while total < target_len:
        line = lines[rng.randrange(0, len(lines))]
        try:
            item = json.loads(line)
        except Exception:
            continue
        parts = []
        inst = (item.get("instruction") or "").strip()
        ctx = (item.get("context") or "").strip()
        if inst:
            parts.append(f"Instruction: {inst}")
        if ctx:
            parts.append(f"Context: {ctx}")
        parts.append("Answer:")
        text = "\n".join(parts) + "\n"
        token_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        if token_ids.numel() == 0:
            continue
        ids_chunks.append(token_ids)
        total += int(token_ids.numel())
    ids = torch.cat(ids_chunks, dim=0)[:target_len]
    if int(ids.numel()) != int(target_len):
        raise RuntimeError(f"Failed to build token sequence of length {target_len} (got {ids.numel()})")
    return ids


def _load_pruners(hidden_size: int, pruning_ckpt: Path, prune_layers: list[int]):
    pruners = torch.nn.ModuleDict({str(i): TokenPruningModule(hidden_size) for i in prune_layers})
    state_dict = torch.load(pruning_ckpt, map_location="cpu")
    pruners.load_state_dict(state_dict, strict=False)
    pruners.half().eval()
    for p in pruners.parameters():
        p.requires_grad = False
    return pruners


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--pruning_ckpt", required=True)
    ap.add_argument("--keep_ratio", type=float, default=0.7)
    ap.add_argument("--prune_layers", type=int, nargs="+", default=None)
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dolly_jsonl", type=str, default=str(PROJECT_ROOT / "data" / "raw" / "databricks-dolly-15k.jsonl"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model_path)
    pruning_ckpt = Path(args.pruning_ckpt)
    dolly_jsonl = Path(args.dolly_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map=None,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    state_dict = torch.load(pruning_ckpt, map_location="cpu")
    avail_layers = sorted({int(k.split(".", 1)[0]) for k in state_dict.keys() if k.split(".", 1)[0].isdigit()})
    prune_layers = list(args.prune_layers) if args.prune_layers else avail_layers

    pruners = _load_pruners(hidden_size=int(model.config.hidden_size), pruning_ckpt=pruning_ckpt, prune_layers=avail_layers).to(device)

    ids = _build_ids(tokenizer, dolly_jsonl, int(args.length), seed=int(args.seed))
    input_ids = ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    def _run(strategy: str, seed: int):
        with torch.no_grad():
            _logits, stats, pruned_input_ids = prefill_with_pruning(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pruning_modules=pruners,
                keep_ratio=float(args.keep_ratio),
                prune_layers=prune_layers,
                return_pruned_input_ids=True,
                logits_mode="last",
                token_select_strategy=strategy,
                random_seed=seed,
            )
        return stats, pruned_input_ids

    base_seed = int(args.seed)
    stats_e, ids_e = _run("erecap", base_seed)
    stats_r1, ids_r1 = _run("random", base_seed)
    stats_r2, ids_r2 = _run("random", base_seed)
    stats_r3, ids_r3 = _run("random", base_seed + 1)
    stats_c, ids_c = _run("recency", base_seed)

    def _layer_lengths(stats):
        return [(s.get("layer"), s.get("original_length"), s.get("tokens_kept")) for s in stats.get("layer_stats", [])]

    print("[Check] final_length:")
    print(f"  erecap  : {stats_e.get('final_length')}")
    print(f"  random  : {stats_r1.get('final_length')}")
    print(f"  recency : {stats_c.get('final_length')}")
    assert int(stats_e.get("final_length")) == int(stats_r1.get("final_length")) == int(stats_c.get("final_length"))

    print("[Check] per-layer lengths (layer, in_len, kept_len):")
    print("  erecap :", _layer_lengths(stats_e))
    print("  random :", _layer_lengths(stats_r1))
    print("  recency:", _layer_lengths(stats_c))
    assert _layer_lengths(stats_e) == _layer_lengths(stats_r1) == _layer_lengths(stats_c)

    print("[Check] random determinism:")
    same = torch.equal(ids_r1.cpu(), ids_r2.cpu())
    diff = not torch.equal(ids_r1.cpu(), ids_r3.cpu())
    print(f"  same_seed_equal: {same}")
    print(f"  diff_seed_diff : {diff}")
    assert same, "Random baseline should be deterministic under the same seed"
    assert diff, "Random baseline should change under a different seed"

    print("[OK] Sanity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

