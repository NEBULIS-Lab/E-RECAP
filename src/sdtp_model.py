from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class SDTPModel(nn.Module):
    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=None,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        for param in self.model.parameters():
            param.requires_grad = False

        self.pruning_modules = nn.ModuleDict()
        self.gumbel_tau = 1.0

    def attach_pruning_module(self, layer_idx: int, module: nn.Module) -> None:
        self.pruning_modules[str(layer_idx)] = module.to(self.device)

    def _extract_keep_scores(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            return logits[..., 0]
        return logits

    def apply_pruning(
        self,
        hidden_states: torch.Tensor,
        scores: torch.Tensor,
        keep_ratio: float,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        keep_k = max(1, int(seq_len * keep_ratio))

        topk = scores.topk(keep_k, dim=-1, largest=True)
        topk_indices = topk.indices.sort(dim=-1).values

        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        pruned_hidden_states = torch.gather(hidden_states, dim=1, index=gather_index)

        pruned_attention_mask = None
        if attention_mask is not None:
            pruned_attention_mask = torch.gather(attention_mask, dim=1, index=topk_indices)

        return pruned_hidden_states, pruned_attention_mask, topk_indices

    def prune_past_key_values(
        self,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        kept_indices: torch.Tensor,
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        if past_key_values is None:
            return None
        pruned_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for key, value in past_key_values:
            pruned_key = key.index_select(dim=2, index=kept_indices.squeeze(0))
            pruned_value = value.index_select(dim=2, index=kept_indices.squeeze(0))
            pruned_cache.append((pruned_key, pruned_value))
        return pruned_cache

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, dtype=torch.long, device=hidden_states.device)
        return self.model._prepare_decoder_attention_mask(
            attention_mask,
            input_shape,
            hidden_states.dtype,
            hidden_states.device,
            past_key_values_length=0,
        )

    def forward_prefill_with_pruning(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        keep_ratio: float,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        else:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.device)

        hidden_states = self.model.model.embed_tokens(input_ids)
        position_ids = torch.arange(hidden_states.size(1), device=self.device).unsqueeze(0)
        extended_attention = self._prepare_attention_mask(attention_mask, hidden_states.shape[:2], hidden_states)

        kept_indices: Dict[int, torch.Tensor] = {}
        past_key_values = None

        for idx, block in enumerate(self.model.model.layers):
            pruning_module = self.pruning_modules.get(str(idx), None)

            if pruning_module is not None:
                pruning_module = pruning_module.to(self.device)
                if self.training:
                    logits = pruning_module(hidden_states)
                    gumbel = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False, dim=-1)
                    keep_prob = gumbel[..., 0].unsqueeze(-1)
                    hidden_states = hidden_states * keep_prob
                else:
                    scores = pruning_module(hidden_states)
                    scores = self._extract_keep_scores(scores)
                    hidden_states, attention_mask, kept = self.apply_pruning(
                        hidden_states, scores, keep_ratio, attention_mask
                    )
                    kept_indices[idx] = kept.detach()
                    position_ids = torch.arange(hidden_states.size(1), device=self.device).unsqueeze(0)
                    extended_attention = self._prepare_attention_mask(attention_mask, hidden_states.shape[:2], hidden_states)
                    past_key_values = self.prune_past_key_values(past_key_values, kept)

            block_outputs = block(
                hidden_states,
                attention_mask=extended_attention,
                position_ids=position_ids,
                past_key_value=past_key_values[idx] if past_key_values is not None else None,
                use_cache=past_key_values is not None,
            )

            hidden_states = block_outputs[0]
            if self.model.config.use_cache:
                if past_key_values is None:
                    past_key_values = [None] * len(self.model.model.layers)
                past_key_values[idx] = block_outputs[1]

        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)

        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.long, device=self.device)

        return logits, kept_indices, attention_mask, hidden_states
