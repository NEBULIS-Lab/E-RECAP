# End-to-End Latency Benchmarking for SDTP
# Measures prefill + decode (128 tokens) latency
# Simplified implementation to avoid 0-length tensor errors

import time
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from inference_sdtp
try:
    from inference_sdtp import (
        load_model_and_pruners,
        prefill_with_pruning,
        KEEP09_CONFIG,
        KEEP08_CONFIG,
        KEEP07_CONFIG,
        PRUNE_LAYERS,
        MIN_HEAD_TOKENS,
        MIN_TAIL_RATIO,
        build_dummy_input,
        device,
    )
    from sdtp_model import SDTPModel
except ImportError:
    # Fallback if running as standalone
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from inference_sdtp import (
        load_model_and_pruners,
        prefill_with_pruning,
        KEEP09_CONFIG,
        KEEP08_CONFIG,
        KEEP07_CONFIG,
        PRUNE_LAYERS,
        MIN_HEAD_TOKENS,
        MIN_TAIL_RATIO,
        build_dummy_input,
        device,
    )
    from sdtp_model import SDTPModel

MAX_NEW_TOKENS = 128  # Generate 128 tokens as in paper


def run_end2end_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict:
    """
    Run baseline end-to-end inference (prefill + generate).
    
    Baseline: no pruning, pure model.generate().
    This function does NOT touch any SDTP modules or pruning logic.
    
    Args:
        model: The language model
        tokenizer: The tokenizer (unused, kept for API consistency)
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        max_new_tokens: Number of tokens to generate
        
    Returns:
        Dictionary with timing information
    """
    # CRITICAL: Guard against 0-length input
    assert input_ids.shape[1] > 0, f"input_ids seq_len must be > 0, got {input_ids.shape[1]}"
    assert attention_mask is None or attention_mask.shape[1] == input_ids.shape[1], \
        f"attention_mask seq_len {attention_mask.shape[1]} != input_ids seq_len {input_ids.shape[1]}"
    
    model.eval()
    
    with torch.no_grad():
        # Warmup: one tiny generate to compile kernels
        warmup_ids = input_ids[:, :min(8, input_ids.shape[1])]
        warmup_mask = attention_mask[:, :min(8, attention_mask.shape[1])] if attention_mask is not None else None
        _ = model.generate(
            input_ids=warmup_ids,
            attention_mask=warmup_mask,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Measure total end2end time with a single generate() call
        # This is the safest approach - no manual KV cache manipulation
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_start = time.perf_counter()
        
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_end = time.perf_counter()
        total_time = total_end - total_start
        
        # For reporting, we approximate:
        # - prefill_latency: time for first forward pass (we measure separately)
        # - decode_latency: total - prefill
        # But to keep things simple and robust, we measure prefill separately
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        
        # Single forward pass to approximate prefill time
        model_inputs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        _ = model(**model_inputs)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        prefill_time = prefill_end - prefill_start
        
        # Decode time is approximate: total - prefill
        decode_time = total_time - prefill_time
        if decode_time < 0:
            decode_time = 0.0  # Safety: ensure non-negative
    
    return {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "kv_lens_after_prefill": [input_ids.shape[1]] * len(model.model.layers),  # Approximate
        "kv_lens_final": [generated.shape[1]] * len(model.model.layers),
        "generated_length": generated.shape[1] - input_ids.shape[1],
    }


def run_end2end_sdtp(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pruning_modules: nn.ModuleDict,
    keep_ratio: float,
    prune_layers: List[int],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict:
    """
    Run SDTP end-to-end inference (prefill with pruning + decode with pruned KV cache).

    Steps:
      1. Wrap the base model in SDTPModel, attach pruning modules.
      2. Run prefill_with_pruning_infer to get:
         - logits
         - pruning_stats
         - pruned past_key_values
         - pruned attention_mask
      3. Run a custom greedy decode loop that uses:
         - last generated token
         - pruned attention_mask
         - pruned past_key_values
      4. Measure prefill_time, decode_time, total_time.
    """
    assert input_ids.shape[1] > 0, f"input_ids seq_len must be > 0, got {input_ids.shape[1]}"
    assert attention_mask is None or attention_mask.shape[1] == input_ids.shape[1], \
        f"attention_mask seq_len {attention_mask.shape[1]} != input_ids seq_len {input_ids.shape[1]}"

    device = input_ids.device
    model.eval()

    # Wrap base model and tokenizer with SDTPModel to reuse inference utilities
    sdtp = SDTPModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # Attach pruning modules
    for name, module in pruning_modules.items():
        layer_idx = int(name)
        sdtp.attach_pruning_module(layer_idx, module)

    with torch.no_grad():
        # Prefill with pruning
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_start = time.perf_counter()

        logits, pruning_stats, past_key_values, pruned_attn = sdtp.prefill_with_pruning_infer(
            input_ids, attention_mask, keep_ratio=keep_ratio
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        prefill_time = prefill_end - prefill_start

        # Sanity checks
        assert logits.size(1) > 0, "SDTP prefill returned empty sequence"
        final_seq_len = pruned_attn.size(1)
        assert final_seq_len == pruning_stats.get("final_length", final_seq_len), \
            "Mismatch between pruned attention length and pruning_stats['final_length']"

        # KV lengths after prefill: we approximate by final_seq_len for all layers
        kv_lens_after_prefill = [final_seq_len] * len(model.model.layers)

        # Decode with pruned KV cache
        # We build a simple greedy decoding loop:
        #   - start from the last token prediction of prefill logits
        #   - at each step, pass it with past_key_values and pruned_attn
        #   - append the predicted token to generated_ids (for inspection only)

        # Start decode from the last token prediction of prefill
        next_token_logits = logits[:, -1, :]  # [1, vocab]
        first_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [1, 1]
        
        generated_ids = first_token.to(device)  # Start with the first generated token
        attn = pruned_attn.to(device)
        kv = past_key_values

        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_start = time.perf_counter()

        for step in range(max_new_tokens - 1):  # -1 because we already generated the first token
            # Use the last generated token as the next input
            next_input = generated_ids[:, -1:].to(device)  # [batch=1, 1]

            outputs = model(
                input_ids=next_input,
                attention_mask=attn,
                past_key_values=kv,
                use_cache=True,
            )
            next_logits = outputs.logits[:, -1, :]  # [1, vocab]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # [1, 1]

            # Append new token to generated_ids (for final length reporting)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Update attention mask and KV cache
            new_mask_token = torch.ones(attn.size(0), 1, dtype=attn.dtype, device=attn.device)
            attn = torch.cat([attn, new_mask_token], dim=1)
            kv = outputs.past_key_values

        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_end = time.perf_counter()
        decode_time = decode_end - decode_start

    total_time = prefill_time + decode_time

    kv_lens_final = [kv_lens_after_prefill[0] + max_new_tokens] * len(model.model.layers)
    generated_length = generated_ids.shape[1]  # Total generated tokens (including first one from prefill)

    return {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "kv_lens_after_prefill": kv_lens_after_prefill,
        "kv_lens_final": kv_lens_final,
        "generated_length": int(generated_length),
        "pruning_stats": pruning_stats,
    }


def run_end2end_latency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    use_sdtp: bool = False,
    pruning_modules: Optional[nn.ModuleDict] = None,
    keep_ratio: float = 0.7,
    prune_layers: Optional[List[int]] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict:
    """
    Run end-to-end latency benchmark.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        input_ids: Input token IDs
        attention_mask: Attention mask
        use_sdtp: Whether to use SDTP pruning
        pruning_modules: Pruning modules (required if use_sdtp=True)
        keep_ratio: Token keep ratio (required if use_sdtp=True)
        prune_layers: List of layers to prune (required if use_sdtp=True)
        max_new_tokens: Number of tokens to generate
        
    Returns:
        Dictionary with timing and KV cache information
    """
    # CRITICAL: Validate inputs before processing
    if input_ids.shape[1] == 0:
        raise ValueError(f"run_end2end_latency: input_ids seq_len is 0!")
    
    if prune_layers is None:
        prune_layers = PRUNE_LAYERS
    
    if use_sdtp:
        if pruning_modules is None:
            raise ValueError("pruning_modules required when use_sdtp=True")
        return run_end2end_sdtp(
            model, tokenizer, input_ids, attention_mask,
            pruning_modules, keep_ratio, prune_layers, max_new_tokens,
        )
    else:
        return run_end2end_baseline(
            model, tokenizer, input_ids, attention_mask, max_new_tokens,
        )


if __name__ == "__main__":
    # Test script
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--use_sdtp", action="store_true")
    parser.add_argument("--config", choices=["keep09", "keep08", "keep07"], default="keep07")
    args = parser.parse_args()
    
    # Load model
    if args.use_sdtp:
        if args.config == "keep09":
            config = KEEP09_CONFIG
        elif args.config == "keep08":
            config = KEEP08_CONFIG
        else:
            config = KEEP07_CONFIG
        
        model, tokenizer, pruners = load_model_and_pruners(prune_layers=config["prune_layers"])
        keep_ratio = config["keep_ratio"]
        prune_layers = config["prune_layers"]
    else:
        model, tokenizer, _ = load_model_and_pruners()
        pruners = None
        keep_ratio = 1.0
        prune_layers = []
    
    # Build input
    input_ids, attention_mask = build_dummy_input(tokenizer, args.length)
    
    # Run benchmark
    result = run_end2end_latency(
        model, tokenizer, input_ids, attention_mask,
        use_sdtp=args.use_sdtp,
        pruning_modules=pruners,
        keep_ratio=keep_ratio,
        prune_layers=prune_layers,
    )
    
    print(f"\n{'='*60}")
    print(f"End2End Benchmark Results (Length: {args.length})")
    print(f"{'='*60}")
    print(f"Prefill time: {result['prefill_time']:.4f}s")
    print(f"Decode time: {result['decode_time']:.4f}s")
    print(f"Total time: {result['total_time']:.4f}s")
    print(f"\nKV lengths after prefill: {result['kv_lens_after_prefill']}")
    if args.use_sdtp and 'pruning_stats' in result:
        print(f"Pruning stats: {result['pruning_stats']}")
