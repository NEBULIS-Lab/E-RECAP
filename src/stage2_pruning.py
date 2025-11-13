import random
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2-7B"
DATA_PATH = "dolly15k"
SAL_PATH = "checkpoints/saliency.pt"
MAX_LEN = 512
BATCH_SIZE = 1
LR = 1e-4
EPOCHS = 2
PRUNE_LAYERS = [4, 7, 10, 13, 16, 19, 22, 25, 28, 31]
KEEP_RATIO = 0.7
TEMPERATURE = 1.0


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


def ranking_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    diff_target = target.unsqueeze(1) - target.unsqueeze(0)
    margin = 1.0
    loss = F.relu(margin - diff_pred * torch.sign(diff_target))
    return loss.mean()


def build_dataloader(tokenizer: AutoTokenizer, dataset, num_examples: int) -> DataLoader:
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:num_examples]

    examples: List[Dict[str, torch.Tensor]] = []
    for idx in indices:
        item = dataset[idx]
        text = f"{item['context']}\n{item['response']}"
        encoding = tokenizer(
            text,
            max_length=MAX_LEN,
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

    return DataLoader(examples, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    hidden_size = model.config.hidden_size
    pruning_modules = nn.ModuleDict()
    for layer_idx in PRUNE_LAYERS:
        pruning_modules[str(layer_idx)] = TokenPruningModule(hidden_size).to(device)

    saliency_data = torch.load(SAL_PATH)
    dataset = load_from_disk(DATA_PATH)["train"]
    num_examples = min(len(next(iter(saliency_data.values()))), len(dataset))
    dataloader = build_dataloader(tokenizer, dataset, num_examples)

    optimizer = torch.optim.Adam(pruning_modules.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            hidden_states = model.model.embed_tokens(input_ids)

            sample_index = batch_idx * BATCH_SIZE

            for layer_idx, block in enumerate(model.model.layers):
                block_outputs = block(hidden_states, attention_mask=attention_mask, use_cache=False)
                hidden_states = block_outputs[0]

                if layer_idx in PRUNE_LAYERS:
                    module = pruning_modules[str(layer_idx)]
                    logits = module(hidden_states.squeeze(0))
                    mask_logits = torch.stack(
                        [logits, torch.zeros_like(logits)], dim=-1
                    )
                    soft_mask = F.gumbel_softmax(
                        mask_logits,
                        tau=TEMPERATURE,
                        hard=False,
                        dim=-1,
                    )[:, 0]
                    hidden_states = hidden_states * soft_mask.unsqueeze(0).unsqueeze(-1)

                    target_list = saliency_data[layer_idx]
                    target_sal = target_list[sample_index + 0].to(device)

                    pred_sal = logits

                    mse = F.mse_loss(pred_sal, target_sal)
                    rank = ranking_loss(pred_sal, target_sal)
                else:
                    mse = torch.tensor(0.0, device=device)
                    rank = torch.tensor(0.0, device=device)

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

    torch.save(pruning_modules.state_dict(), "checkpoints/pruning_module.pt")


if __name__ == "__main__":
    main()
