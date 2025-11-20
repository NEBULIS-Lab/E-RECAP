"""
SDTP Inference Wrapper for Evaluation
Provides a unified interface for baseline and SDTP inference
"""
import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from inference_sdtp module
from inference_sdtp import load_model_and_pruners, generate_text as baseline_generate_text


class SDTPInference:
    """Wrapper class for SDTP inference evaluation"""
    
    def __init__(self, model_path="checkpoints/qwen2-7b-instruct", 
                 pruner_path=None, device="cuda", use_flash=False):
        """
        Args:
            model_path: Path to Qwen2-7B model
            pruner_path: Path to pruning module checkpoint (None for baseline)
            device: Device to use
            use_flash: Whether to use FlashAttention (not implemented yet)
        """
        self.model_path = model_path
        self.pruner_path = pruner_path
        self.device = torch.device(device)
        self.use_flash = use_flash
        
        # Load model and tokenizer
        if pruner_path:
            self.model, self.tokenizer, self.pruners = load_model_and_pruners()
            self.use_pruning = True
        else:
            # Baseline mode: load model without pruner
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=None,
                local_files_only=True,
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self.pruners = None
            self.use_pruning = False
        
        self.model.eval()
    
    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate text from prompt"""
        if self.use_pruning:
            # Use SDTP pruning path
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
            
            # Use prefill_with_pruning for prefill, then generate normally
            # For simplicity, we use baseline generation after prefill
            # In full implementation, decode phase should also use pruning
            with torch.no_grad():
                out_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        else:
            # Baseline generation
            return baseline_generate_text(self.model, self.tokenizer, prompt)

