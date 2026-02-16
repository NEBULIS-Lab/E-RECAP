"""Stage 1: compute gradient×hidden saliency supervision for pruning.

This script was originally hard-coded for Qwen2-7B + Dolly-15k. It now supports
CLI arguments so we can:
- switch backbones (different layer counts)
- change pruning layers
- change data/model paths
"""

import argparse
import random
from typing import Any, List, Optional

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# Defaults (kept compatible with older notes)
DEFAULT_MODEL_PATH = "checkpoints/qwen2-7b-instruct"
DEFAULT_DATA_PATH = "dolly15k"
DEFAULT_PRUNE_LAYERS = [4, 7, 10, 13, 16, 19, 22, 25]

MAX_LEN = 512
BATCH_SIZE = 1

def _parse_layers(values: Optional[List[str]], default: List[int]) -> List[int]:
    if not values:
        return list(default)
    out: List[int] = []
    for v in values:
        v = str(v).strip()
        if not v:
            continue
        out.append(int(v))
    return out


def build_dataloader(tokenizer: AutoTokenizer, dataset, indices: List[int]) -> DataLoader:

    examples = []
    for idx in indices:
        item = dataset[int(idx)]
        # Include instruction, context, and response for complete semantic information
        parts = [item.get('instruction', ''), item.get('context', ''), item['response']]
        text = '\n'.join([p for p in parts if p])  # Only join non-empty parts
        enc = tokenizer(
            text,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        examples.append(
            {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }
        )
    def collate_fn(batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return DataLoader(examples, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--num_samples", type=int, default=1000)
    # Support both legacy --out_path and scripts/run_stage1.sh's --output_path
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="checkpoints/saliency.pt")
    parser.add_argument("--prune_layers", nargs="+", default=None, help="Space-separated layer indices")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prune_layers = _parse_layers(args.prune_layers, default=DEFAULT_PRUNE_LAYERS)
    out_path = args.out_path if args.out_path else args.output_path

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    # Some backbones (e.g., Mistral/LLaMA) ship without a pad token.
    # We need padding for fixed-length batching in Stage1/Stage2.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    # for p in model.parameters():
        # p.requires_grad = False

    dataset = load_from_disk(args.data_path)["train"]
    indices = list(range(len(dataset)))
    random.seed(int(args.seed))
    random.shuffle(indices)
    indices = indices[: int(args.num_samples)]
    dataloader = build_dataloader(tokenizer, dataset, indices)

    forward_cache = {}
    backward_cache = {}
    saliency_results = {k: [] for k in prune_layers}

    def create_hooks(layer_idx):
        def forward_hook(_module, _inp, out):
            hidden = out[0] if isinstance(out, (tuple, list)) else out
            forward_cache[layer_idx] = hidden.detach()

        def backward_hook(_module, grad_in, grad_out):
            grad_hidden = grad_out[0] if isinstance(grad_out, (tuple, list)) else grad_out
            backward_cache[layer_idx] = grad_hidden.detach()

        return forward_hook, backward_hook

    hooks = []
    try:
        for idx in prune_layers:
            layer = model.model.layers[idx]
            f_hook, b_hook = create_hooks(idx)
            hooks.append(layer.register_forward_hook(f_hook))
            hooks.append(layer.register_full_backward_hook(b_hook))

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            model.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                use_cache=False,
            )
            loss = outputs.loss
            loss.backward()

            for idx in prune_layers:
                hidden = forward_cache.get(idx)
                grad = backward_cache.get(idx)
                if hidden is None or grad is None:
                    continue

                sal = (hidden * grad).sum(dim=-1)
                saliency_results[idx].append(sal.float().cpu().squeeze(0))

            forward_cache.clear()
            backward_cache.clear()

    finally:
        for h in hooks:
            h.remove()

    payload: dict[str, Any] = {
        "__meta__": {
            "model_path": args.model_path,
            "data_path": args.data_path,
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "prune_layers": prune_layers,
            "indices": indices,
            "max_len": MAX_LEN,
        }
    }
    payload.update(saliency_results)
    torch.save(payload, out_path)
    print(f"[OK] Saliency saved to {out_path}")


if __name__ == "__main__":
    main()
