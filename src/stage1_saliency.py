import random
from typing import Dict, List

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2-7B"
DATA_PATH = "dolly15k"
NUM_EXAMPLES = 1000
MAX_LEN = 512
BATCH_SIZE = 1
PRUNE_LAYERS = [4, 7, 10, 13, 16, 19, 22, 25, 28, 31]


def build_dataloader(tokenizer: AutoTokenizer) -> DataLoader:
    dataset = load_from_disk(DATA_PATH)["train"]
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:NUM_EXAMPLES]

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

    for param in model.parameters():
        param.requires_grad = False

    dataloader = build_dataloader(tokenizer)

    forward_cache: Dict[int, torch.Tensor] = {}
    backward_cache: Dict[int, torch.Tensor] = {}

    saliency_results: Dict[int, List[torch.Tensor]] = {layer: [] for layer in PRUNE_LAYERS}

    def create_hooks(layer_idx: int):
        def forward_hook(_module, _input, output):
            hidden_states = output[0] if isinstance(output, (tuple, list)) else output
            forward_cache[layer_idx] = hidden_states.detach()

        def backward_hook(_module, grad_input, grad_output):
            grad_hidden = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
            backward_cache[layer_idx] = grad_hidden.detach()

        return forward_hook, backward_hook

    hooks: List[torch.utils.hooks.RemovableHandle] = []
    try:
        for layer_idx in PRUNE_LAYERS:
            layer = model.model.layers[layer_idx]
            f_hook, b_hook = create_hooks(layer_idx)
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

            for layer_idx in PRUNE_LAYERS:
                hidden_states = forward_cache.get(layer_idx)
                grad_states = backward_cache.get(layer_idx)
                if hidden_states is None or grad_states is None:
                    continue

                saliency = (hidden_states * grad_states).sum(dim=-1)
                saliency_results[layer_idx].append(saliency.float().cpu().squeeze(0))

            forward_cache.clear()
            backward_cache.clear()

    finally:
        for hook in hooks:
            hook.remove()

    torch.save(saliency_results, "checkpoints/saliency.pt")


if __name__ == "__main__":
    main()
