"""Stage 2: train lightweight TokenPruningModule(s) using Stage 1 saliency supervision.

This script now supports CLI arguments (used by `scripts/run_stage2.sh`) so we can:
- switch backbones (different hidden sizes / layer counts)
- change pruning layer sets
- avoid overwriting checkpoints by selecting output paths
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = "checkpoints/qwen2-7b-instruct"
DEFAULT_DATA_PATH = "dolly15k"
DEFAULT_SAL_PATH = "checkpoints/saliency.pt"
DEFAULT_OUT_PATH = "checkpoints/pruning_module.pt"
DEFAULT_PRUNE_LAYERS = [4, 7, 10, 13, 16, 19, 22, 25]
DEFAULT_MAX_LEN = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 2
DEFAULT_TEMPERATURE = 1.0


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


# =======================
# Token Pruning Module
# =======================
class TokenPruningModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),  # Use GELU as in paper (was ReLU)
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states):
        logits = self.scorer(hidden_states).squeeze(-1)
        return logits


# =======================
# Ranking loss (logistic loss form as in paper)
# =======================
def ranking_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Logistic ranking loss as in the paper:
    L_r = sum_{i<j} log(1 + exp(-(π_i - π_j) * sign(π̂_i - π̂_j)))
    """
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)  # [N, N]
    diff_target = target.unsqueeze(1) - target.unsqueeze(0)  # [N, N]
    sign_target = torch.sign(diff_target)
    # Logistic loss: log(1 + exp(-diff_pred * sign_target))
    loss = torch.log(1 + torch.exp(-diff_pred * sign_target))
    return loss.mean()


# =======================
# Data loader
# =======================
def build_dataloader(
    tokenizer: AutoTokenizer,
    dataset,
    indices: List[int],
    *,
    max_len: int,
    batch_size: int,
) -> DataLoader:
    examples: List[Dict[str, torch.Tensor]] = []
    for idx in indices:
        item = dataset[int(idx)]
        # Include instruction, context, and response for complete semantic information
        parts = [item.get('instruction', ''), item.get('context', ''), item['response']]
        text = '\n'.join([p for p in parts if p])  # Only join non-empty parts
        encoding = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        examples.append(
            {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }
        )

    def collate_fn(batch):
        input_ids = torch.stack([example["input_ids"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# =======================
# Main training loop
# =======================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--saliency_path", type=str, default=DEFAULT_SAL_PATH)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUT_PATH)
    parser.add_argument("--prune_layers", nargs="+", default=None, help="Space-separated layer indices")
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # This training code assumes batch_size=1 because saliency targets are stored per-sample.
    if int(args.batch_size) != 1:
        raise ValueError("stage2_pruning.py currently supports only --batch_size 1 (saliency alignment)")

    # Load saliency payload (new format may include __meta__ with indices and prune_layers)
    raw_sal = torch.load(args.saliency_path, map_location="cpu")
    meta = None
    if isinstance(raw_sal, dict) and "__meta__" in raw_sal:
        meta = raw_sal.get("__meta__") or {}
        raw_sal = {k: v for k, v in raw_sal.items() if k != "__meta__"}
    saliency_data: dict = raw_sal  # layer_idx -> list[tensor]

    # Determine pruning layers (CLI overrides meta; otherwise use meta; else defaults)
    meta_layers = meta.get("prune_layers") if isinstance(meta, dict) else None
    prune_layers = _parse_layers(args.prune_layers, default=_parse_layers(meta_layers, DEFAULT_PRUNE_LAYERS))

    # Load model (frozen)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True, use_fast=False)
    # Some backbones (e.g., Mistral/LLaMA) ship without a pad token.
    # We need padding for fixed-length batching and for ignore_index in LM loss.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    # Freeze Qwen
    for p in model.parameters():
        p.requires_grad = False

    # Init pruning modules
    hidden_size = model.config.hidden_size
    pruning_modules = nn.ModuleDict()
    for layer_idx in prune_layers:
        pruning_modules[str(layer_idx)] = TokenPruningModule(hidden_size).to(device)

    # <<< NEW: convert pruning modules to fp16 >>>
    pruning_modules.half()

    dataset = load_from_disk(args.data_path)["train"]

    # Align dataset samples with Stage1 indices when available
    indices = None
    if isinstance(meta, dict) and meta.get("indices") is not None:
        indices = [int(i) for i in meta.get("indices")]
    else:
        # Backward-compatibility: old saliency.pt has no indices, so we cannot guarantee alignment
        # (this preserves previous behavior but results may be noisier).
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        print("[Warn] saliency checkpoint has no __meta__/indices; Stage2 sample alignment is not guaranteed.")

    # Determine how many examples we can train on
    # (limited by both dataset size and saliency list lengths)
    sal_len = min(len(saliency_data.get(l, [])) for l in prune_layers)
    num_examples = min(len(indices), len(dataset), sal_len)
    indices = indices[:num_examples]

    dataloader = build_dataloader(
        tokenizer,
        dataset,
        indices,
        max_len=int(args.max_len),
        batch_size=int(args.batch_size),
    )

    optimizer = torch.optim.Adam(pruning_modules.parameters(), lr=float(args.learning_rate))

    # =======================
    # Training
    # =======================
    for epoch in range(int(args.epochs)):
        print(f"[Epoch {epoch+1}/{int(args.epochs)}]")
        for batch_idx, batch in enumerate(dataloader):

            input_ids = batch["input_ids"].to(device)
            attention_mask_2d = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # embedding
            hidden_states = model.model.embed_tokens(input_ids)

            # Build 4D attention mask: causal + padding (batch, 1, q_len, kv_len).
            # Many decoder-layer implementations expect an additive attention bias.
            seq_len = int(hidden_states.size(1))
            attn_dtype = hidden_states.dtype
            causal_mask = torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=device,
                dtype=attn_dtype,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)  # [1,1,L,L]
            padding_mask = torch.zeros(
                (attention_mask_2d.size(0), 1, 1, seq_len),
                device=device,
                dtype=attn_dtype,
            )
            padding_mask.masked_fill_(attention_mask_2d[:, None, None, :].eq(0), float("-inf"))
            attention_mask_4d = causal_mask + padding_mask

            sample_index = batch_idx * int(args.batch_size)

            # forward through each layer
            for layer_idx, block in enumerate(model.model.layers):

                # mandatory: position_ids
                position_ids = torch.arange(
                    0, hidden_states.size(1),
                    dtype=torch.long,
                    device=device
                ).unsqueeze(0)

                block_outputs = block(
                    hidden_states,
                    attention_mask=attention_mask_4d,   # DO NOT pass 2D mask
                    position_ids=position_ids,
                    use_cache=False,
                )
                hidden_states = block_outputs[0]

                if layer_idx in prune_layers:
                    module = pruning_modules[str(layer_idx)]

                    logits = module(hidden_states.squeeze(0))

                    mask_logits = torch.stack(
                        [logits, torch.zeros_like(logits)],
                        dim=-1,
                    )
                    soft_mask = F.gumbel_softmax(
                        mask_logits,
                        tau=float(args.temperature),
                        hard=False,
                        dim=-1,
                    )[:, 0]
                    hidden_states = hidden_states * soft_mask.unsqueeze(0).unsqueeze(-1)

                    target_list = saliency_data[layer_idx]
                    target_sal = target_list[sample_index].to(device)

                    pred_sal = logits
                    mse = F.mse_loss(pred_sal, target_sal)
                    rank = ranking_loss(pred_sal, target_sal)
                else:
                    mse = torch.tensor(0.0, device=device)
                    rank = torch.tensor(0.0, device=device)

            # LM loss
            logits = model.lm_head(hidden_states)
            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )

            loss = lm_loss + mse + rank
            loss.backward()
            optimizer.step()

    os.makedirs(str(Path(args.output_path).parent), exist_ok=True)
    torch.save(pruning_modules.state_dict(), args.output_path)
    print(f"[OK] pruning_module.pt saved to {args.output_path}")


if __name__ == "__main__":
    main()
