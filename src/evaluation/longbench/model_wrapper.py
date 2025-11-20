"""
Model wrapper for Baseline & SDTP (no actual model loading)
Unified interface for evaluation framework
"""
from typing import Optional


class ModelWrapper:
    """
    Unified wrapper for Baseline & SDTP model.
    
    Does NOT perform inference in this stage.
    Only prepares the structure for future evaluation.
    """

    def __init__(self, model_name: str, pruning_module_path: Optional[str] = None):
        """
        Initialize model wrapper (no actual loading).
        
        Args:
            model_name: Model name or path (e.g., "Qwen/Qwen2-7B" or "checkpoints/qwen2-7b-instruct")
            pruning_module_path: Path to pruning module checkpoint (None for baseline)
        """
        self.model_name = model_name
        self.pruning_module_path = pruning_module_path
        
        # Placeholder fields (not actually loaded)
        self.model = None
        self.tokenizer = None
        self.token_pruner = None
        self.is_loaded = False

    def load_model(self):
        """
        Only prepare the loading process — NO real loading here.
        
        This method is a placeholder that will be implemented
        in the actual evaluation phase.
        """
        print(f"[Init] Preparing model loading (not executed): {self.model_name}")
        
        if self.pruning_module_path:
            print(f"[Init] Pruning module: {self.pruning_module_path}")
            print(f"[Init] Mode: SDTP (with pruning)")
        else:
            print(f"[Init] Mode: Baseline (no pruning)")
        
        print("[Init] Model loading is disabled in setup stage.")
        self.is_loaded = False

    def infer(self, prompt: str, max_new_tokens: int = 128) -> str:
        """
        Placeholder inference — not executed.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text (placeholder)
            
        Raises:
            NotImplementedError: Always, since inference is disabled
        """
        raise NotImplementedError(
            "Inference is disabled in setup stage. "
            "This method will be implemented in the actual evaluation phase."
        )
    
    def is_sdtp(self) -> bool:
        """Check if this wrapper is configured for SDTP"""
        return self.pruning_module_path is not None

