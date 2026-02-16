#!/usr/bin/env python3
"""
Tune / compare pruning-layer placements for E-RECAP.

This script is designed to support the paper claim that pruning layer placement
can be selected via a small held-out calibration set (speed–quality objective).

Important constraints:
- The pruning checkpoint defines which layers are supported (it contains one
  TokenPruningModule per pruning layer). Candidate layer sets are restricted to
  subsets of those available layers unless you train a new checkpoint.
- Different LLM backbones may have different layer counts (e.g., 28 vs 32).
  The script reads model config and validates candidate layer indices.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import gc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_latency(fn, *args, warmup: int = 1, runs: int = 3) -> float:
    for _ in range(warmup):
        _ = fn(*args)
        _cuda_sync()
    times: list[float] = []
    for _ in range(runs):
        _cuda_sync()
        t0 = time.perf_counter()
        _ = fn(*args)
        _cuda_sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / max(1, len(times))


def _parse_available_layers_from_ckpt(state_dict: dict[str, torch.Tensor]) -> list[int]:
    layers: set[int] = set()
    for k in state_dict.keys():
        # e.g. "4.scorer.0.weight" -> layer "4"
        first = k.split(".", 1)[0]
        if first.isdigit():
            layers.add(int(first))
    return sorted(layers)


def _infer_num_layers(model) -> int:
    # Most HF causal LMs expose config.num_hidden_layers
    n = getattr(model.config, "num_hidden_layers", None)
    if isinstance(n, int) and n > 0:
        return n
    # fallback heuristics
    for attr in ("n_layer", "num_layers", "n_layers"):
        v = getattr(model.config, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    raise ValueError("Cannot infer num layers from model config")


def _baseline_last_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Memory-safe baseline prefill that computes logits only for the last position.

    Uses the backbone forward (model.model) to avoid allocating a full [L, V] logits tensor.
    """
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden_states = outputs[0]
        hidden_states = model.model.norm(hidden_states)
        logits_last = model.lm_head(hidden_states[:, -1:, :])
    return logits_last

# --- Import pruning forward from src/inference_erecap.py (reuse canonical implementation)
import sys as _sys

_sys.path.insert(0, str(PROJECT_ROOT / "src"))
from inference_erecap import TokenPruningModule, prefill_with_pruning  # noqa: E402


def _load_pruners(hidden_size: int, pruning_ckpt: Path, prune_layers: list[int]) -> nn.ModuleDict:
    pruning_modules = nn.ModuleDict({str(i): TokenPruningModule(hidden_size) for i in prune_layers})
    state_dict = torch.load(pruning_ckpt, map_location="cpu")
    missing, unexpected = pruning_modules.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading pruner ckpt: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        # allow extra keys (e.g., ckpt has more layers than prune_layers), but warn
        print(f"[Warn] Ignoring unexpected pruner keys (ckpt has extra layers): {len(unexpected)} keys")
    pruning_modules.half().eval()
    for p in pruning_modules.parameters():
        p.requires_grad = False
    return pruning_modules


def _build_token_sequence_from_dolly(
    tokenizer,
    dolly_jsonl: Path,
    target_len: int,
    seed: int,
) -> torch.Tensor:
    rng = random.Random(seed)
    # Read a limited number of lines for speed; shuffle order by sampling indices.
    # The raw file is ~15k lines, so reading all is fine; but keep it lightweight.
    lines = dolly_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise RuntimeError(f"Empty dolly jsonl: {dolly_jsonl}")

    ids_chunks: list[torch.Tensor] = []
    total = 0
    # Keep concatenating random examples until we exceed target_len
    while total < target_len:
        line = lines[rng.randrange(0, len(lines))]
        try:
            item = json.loads(line)
        except Exception:
            continue
        # Build a prompt-like text (no response) to approximate planning contexts
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
    if ids.numel() != target_len:
        raise RuntimeError(f"Failed to build token sequence of length {target_len} (got {ids.numel()})")
    return ids


def _kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    # p, q: [vocab]
    p = torch.log_softmax(p_logits, dim=-1)
    q = torch.log_softmax(q_logits, dim=-1)
    p_prob = torch.softmax(p_logits, dim=-1)
    return torch.sum(p_prob * (p - q)).item()


@dataclass
class CandidateResult:
    prune_layers: list[int]
    avg_speedup: float
    per_length: dict[int, dict[str, float]]
    avg_kl_last: float | None
    top1_match: float | None


def _generate_candidates_pool_spacing(
    pool: list[int],
    num_points_list: list[int],
    max_step: int,
    max_candidates: int,
) -> list[list[int]]:
    m = len(pool)
    cands: list[list[int]] = []
    for num_points in num_points_list:
        if num_points <= 0:
            continue
        for step in range(1, max_step + 1):
            for start in range(0, m):
                idxs = [start + i * step for i in range(num_points)]
                if idxs[-1] >= m:
                    continue
                layers = [pool[i] for i in idxs]
                cands.append(layers)
                if len(cands) >= max_candidates:
                    return _dedup_sorted_candidates(cands)
    return _dedup_sorted_candidates(cands)


def _dedup_sorted_candidates(cands: Iterable[list[int]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for c in cands:
        key = tuple(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(c))
    # prefer longer (more pruning points) first, then earlier start
    out.sort(key=lambda x: (-len(x), x[0] if x else 10**9))
    return out


def _validate_candidate_layers(cand: list[int], num_layers: int, available: set[int]) -> None:
    if not cand:
        raise ValueError("Empty candidate layer list")
    if any((l < 0 or l >= num_layers) for l in cand):
        raise ValueError(f"Candidate has out-of-range layers for num_layers={num_layers}: {cand}")
    if any(l not in available for l in cand):
        raise ValueError(f"Candidate uses layers not present in pruning ckpt: {cand}")
    if sorted(cand) != cand:
        raise ValueError(f"Candidate must be sorted ascending: {cand}")
    if len(set(cand)) != len(cand):
        raise ValueError(f"Candidate has duplicates: {cand}")


def main() -> int:
    # Helps reduce CUDA allocator fragmentation on long runs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="Local HF model directory")
    ap.add_argument("--pruning_ckpt", type=str, required=True, help="Path to pruning_module.pt")
    ap.add_argument("--keep_ratio", type=float, default=0.7)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--num_prompts", type=int, default=3, help="Number of token sequences per length (latency)")
    ap.add_argument("--quality_prompts", type=int, default=2, help="Number of sequences for quality proxy")
    ap.add_argument("--prompt_source", choices=["dolly"], default="dolly")
    ap.add_argument("--dolly_jsonl", type=str, default=str(PROJECT_ROOT / "data" / "raw" / "databricks-dolly-15k.jsonl"))
    ap.add_argument("--seed", type=int, default=0)

    # candidate generation (pool-based)
    ap.add_argument("--num_points", type=int, nargs="+", default=[8, 6, 4])
    ap.add_argument("--max_step", type=int, default=2)
    ap.add_argument("--max_candidates", type=int, default=24)

    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "results" / "layer_placement"))
    args = ap.parse_args()

    device = _device()
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    model_path = Path(args.model_path)
    pruning_ckpt = Path(args.pruning_ckpt)
    dolly_jsonl = Path(args.dolly_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pruning_ckpt.exists():
        raise FileNotFoundError(pruning_ckpt)
    if args.prompt_source == "dolly" and not dolly_jsonl.exists():
        raise FileNotFoundError(dolly_jsonl)

    # Load model/tokenizer
    print(f"[Load] model={model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map=None,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    num_layers = _infer_num_layers(model)
    hidden_size = getattr(model.config, "hidden_size", None)
    print(f"[Model] num_layers={num_layers}, hidden_size={hidden_size}")

    # Load pruning ckpt and derive available layers
    state_dict = torch.load(pruning_ckpt, map_location="cpu")
    avail_layers = _parse_available_layers_from_ckpt(state_dict)
    avail_set = set(avail_layers)
    print(f"[Pruner] ckpt={pruning_ckpt} (layers={avail_layers})")

    # Validate: pruning layers must exist in model
    if any(l >= num_layers for l in avail_layers):
        bad = [l for l in avail_layers if l >= num_layers]
        raise RuntimeError(f"Pruner ckpt references layers >= num_layers ({num_layers}): {bad}")

    pruners = _load_pruners(hidden_size=hidden_size, pruning_ckpt=pruning_ckpt, prune_layers=avail_layers).to(device)

    # Build prompt token sequences (deterministic)
    rng = random.Random(args.seed)
    max_len = max(args.lengths)
    # Build a pool of sequences once, then reuse slices
    seq_pool: dict[int, list[torch.Tensor]] = {}
    for L in sorted(set(args.lengths)):
        seqs: list[torch.Tensor] = []
        for i in range(args.num_prompts):
            seq_ids = _build_token_sequence_from_dolly(tokenizer, dolly_jsonl, L, seed=rng.randrange(0, 10**9))
            seqs.append(seq_ids)
        seq_pool[L] = seqs

    qual_pool: dict[int, list[torch.Tensor]] = {}
    for L in sorted(set(args.lengths)):
        seqs: list[torch.Tensor] = []
        for i in range(args.quality_prompts):
            seq_ids = _build_token_sequence_from_dolly(tokenizer, dolly_jsonl, L, seed=rng.randrange(0, 10**9))
            seqs.append(seq_ids)
        qual_pool[L] = seqs

    # Precompute baseline latencies per length (averaged over sequences)
    baseline_lat: dict[int, float] = {}
    baseline_logits_last: dict[int, list[torch.Tensor]] = {}
    for L, seqs in seq_pool.items():
        latencies: list[float] = []
        for ids in seqs:
            input_ids = ids.unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            lat = _measure_latency(_baseline_last_logits, model, input_ids, attention_mask, warmup=args.warmup, runs=args.runs)
            latencies.append(lat)
        baseline_lat[L] = sum(latencies) / len(latencies)
        print(f"[Baseline] L={L}: {baseline_lat[L]:.4f}s (avg over {len(latencies)} seqs)")

    # Precompute baseline logits (quality proxy) per length
    for L, seqs in qual_pool.items():
        outs: list[torch.Tensor] = []
        for ids in seqs:
            input_ids = ids.unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            with torch.no_grad():
                logits = _baseline_last_logits(model, input_ids, attention_mask)
            outs.append(logits[0, -1, :].float().cpu())
        baseline_logits_last[L] = outs

    # Generate candidates from available layer pool
    candidates = _generate_candidates_pool_spacing(
        pool=avail_layers,
        num_points_list=list(args.num_points),
        max_step=args.max_step,
        max_candidates=args.max_candidates,
    )

    # Ensure the "full pool" candidate exists first
    if avail_layers not in candidates:
        candidates = [avail_layers] + candidates
    candidates = _dedup_sorted_candidates(candidates)

    print(f"[Candidates] {len(candidates)} candidates to evaluate")

    results: list[CandidateResult] = []
    for idx, cand in enumerate(candidates):
        _validate_candidate_layers(cand, num_layers=num_layers, available=avail_set)
        print(f"\n== Candidate {idx+1}/{len(candidates)}: layers={cand} ==")

        per_len: dict[int, dict[str, float]] = {}
        speedups: list[float] = []
        failed = False
        for L, seqs in seq_pool.items():
            pruned_lats: list[float] = []
            pruned_final_lengths: list[int] = []
            for ids in seqs:
                input_ids = ids.unsqueeze(0).to(device)
                attention_mask = torch.ones_like(input_ids, device=device)
                try:
                    def _prefill_once():
                        _logits, stats = prefill_with_pruning(
                            model=model,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pruning_modules=pruners,
                            keep_ratio=float(args.keep_ratio),
                            prune_layers=cand,
                            return_pruned_input_ids=False,
                            logits_mode="last",
                        )
                        return stats

                    lat = _measure_latency(_prefill_once, warmup=args.warmup, runs=args.runs)
                    pruned_lats.append(lat)

                    # One extra call to extract final length (not timed)
                    with torch.no_grad():
                        _logits, stats = prefill_with_pruning(
                            model=model,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pruning_modules=pruners,
                            keep_ratio=float(args.keep_ratio),
                            prune_layers=cand,
                            return_pruned_input_ids=False,
                            logits_mode="last",
                        )
                    pruned_final_lengths.append(int(stats.get("final_length", -1)))
                    del _logits, stats
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        failed = True
                        per_len[L] = {"error": 1.0}  # placeholder numeric key for JSON schema stability
                        print(f"[OOM] Candidate layers={cand} failed at L={L}: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        break
                    raise

            if failed:
                break

            pruned_lat = sum(pruned_lats) / max(1, len(pruned_lats))
            base_lat = baseline_lat[L]
            speedup = base_lat / pruned_lat if pruned_lat > 0 else float("inf")
            speedups.append(speedup)
            per_len[L] = {
                "baseline_s": base_lat,
                "erecap_s": pruned_lat,
                "speedup": speedup,
                "final_len_mean": sum(pruned_final_lengths) / len(pruned_final_lengths),
            }
            print(
                f"[L={L}] baseline={base_lat:.4f}s, erecap={pruned_lat:.4f}s, "
                f"speedup={speedup:.2f}x, final_len~{per_len[L]['final_len_mean']:.0f}"
            )

        avg_speedup = sum(speedups) / len(speedups) if (speedups and not failed) else 0.0

        # Quality proxy: compare last-position logits (baseline vs pruned)
        kls: list[float] = []
        top1_matches = 0
        total = 0
        avg_kl = None
        top1_match = None
        if not failed:
            for L, seqs in qual_pool.items():
                for j, ids in enumerate(seqs):
                    input_ids = ids.unsqueeze(0).to(device)
                    attention_mask = torch.ones_like(input_ids, device=device)
                    try:
                        with torch.no_grad():
                            pruned_logits, _stats = prefill_with_pruning(
                                model=model,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                pruning_modules=pruners,
                                keep_ratio=float(args.keep_ratio),
                                prune_layers=cand,
                                return_pruned_input_ids=False,
                                logits_mode="last",
                            )
                        base_last = baseline_logits_last[L][j]
                        pruned_last = pruned_logits[0, -1, :].float().cpu()
                        kls.append(_kl_divergence(base_last, pruned_last))

                        if int(torch.argmax(base_last)) == int(torch.argmax(pruned_last)):
                            top1_matches += 1
                        total += 1
                        del pruned_logits, _stats
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            failed = True
                            print(f"[OOM] Candidate layers={cand} failed during quality eval at L={L}: {e}")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                            break
                        raise
                if failed:
                    break

            avg_kl = sum(kls) / len(kls) if (kls and not failed) else None
            top1_match = (top1_matches / total) if (total > 0 and not failed) else None
        if avg_kl is not None and top1_match is not None:
            print(f"[Quality] avg_KL(last)={avg_kl:.4f}, top1_match={top1_match:.2%} ({top1_matches}/{total})")

        results.append(
            CandidateResult(
                prune_layers=cand,
                avg_speedup=avg_speedup,
                per_length=per_len,
                avg_kl_last=avg_kl,
                top1_match=top1_match,
            )
        )

        # Best-effort memory cleanup between candidates
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Pick "best" by speedup * top1_match (simple)
    def _score(r: CandidateResult) -> float:
        if r.top1_match is None:
            return r.avg_speedup
        return r.avg_speedup * float(r.top1_match)

    best = None
    if results:
        scored = [r for r in results if r.avg_speedup > 0 and r.top1_match is not None]
        best = max(scored, key=_score) if scored else max(results, key=_score)

    out = {
        "meta": {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": str(model_path),
            "pruning_ckpt": str(pruning_ckpt),
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "keep_ratio": args.keep_ratio,
            "lengths": args.lengths,
            "num_prompts": args.num_prompts,
            "quality_prompts": args.quality_prompts,
            "available_pruner_layers": avail_layers,
            "seed": args.seed,
        },
        "results": [
            {
                "prune_layers": r.prune_layers,
                "avg_speedup": r.avg_speedup,
                "avg_kl_last": r.avg_kl_last,
                "top1_match": r.top1_match,
                "per_length": r.per_length,
                "score": _score(r),
            }
            for r in results
        ],
        "best": {
            "prune_layers": best.prune_layers,
            "avg_speedup": best.avg_speedup,
            "avg_kl_last": best.avg_kl_last,
            "top1_match": best.top1_match,
            "per_length": best.per_length,
            "score": _score(best),
        }
        if best is not None
        else None,
    }

    model_tag = model_path.name.replace("/", "_")
    out_json = out_dir / f"layer_placement_search__{model_tag}__r{args.keep_ratio}__{_utc_now_compact()}.json"
    out_md = out_json.with_suffix(".md")
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write a compact markdown summary table
    lines = []
    lines.append(f"# Layer placement search ({model_tag})")
    lines.append("")
    lines.append(f"- time_utc: `{out['meta']['time_utc']}`")
    lines.append(f"- keep_ratio: `{args.keep_ratio}`")
    lines.append(f"- lengths: `{args.lengths}`")
    lines.append(f"- available pruner layers: `{avail_layers}`")
    lines.append("")
    if out["best"] is not None:
        lines.append("## Best (by speedup × top1_match)")
        lines.append(f"- layers: `{out['best']['prune_layers']}`")
        lines.append(f"- avg_speedup: `{out['best']['avg_speedup']:.3f}`")
        if out["best"]["avg_kl_last"] is not None:
            lines.append(f"- avg_KL(last): `{out['best']['avg_kl_last']:.4f}`")
        if out["best"]["top1_match"] is not None:
            lines.append(f"- top1_match: `{out['best']['top1_match']:.2%}`")
        lines.append("")
    lines.append("## All candidates")
    lines.append("| prune_layers | avg_speedup | avg_KL(last) | top1_match | score |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in out["results"]:
        kl = r["avg_kl_last"]
        tm = r["top1_match"]
        lines.append(
            "| "
            + f"`{r['prune_layers']}` | "
            + f"{r['avg_speedup']:.3f} | "
            + (f"{kl:.4f}" if isinstance(kl, (float, int)) else "NA")
            + " | "
            + (f"{tm:.2%}" if isinstance(tm, (float, int)) else "NA")
            + " | "
            + f"{r['score']:.3f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[OK] Wrote: {out_json}")
    print(f"[OK] Wrote: {out_md}")
    if best is not None:
        print(f"[Best] layers={best.prune_layers} avg_speedup={best.avg_speedup:.3f} top1_match={best.top1_match}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
