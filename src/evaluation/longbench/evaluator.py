"""
LongBench evaluator (no actual inference execution)
Unified evaluation class for LongBench tasks
"""
from typing import Dict, List, Optional
from .dataset import LongBenchDataset
from .model_wrapper import ModelWrapper


class LongBenchEvaluator:
    """
    Evaluate SDTP on LongBench tasks.
    
    In this setup phase, ONLY structure — no inference.
    """

    def __init__(self, task_path: str, model: ModelWrapper):
        """
        Initialize evaluator.
        
        Args:
            task_path: Path to LongBench task JSON file
            model: ModelWrapper instance (baseline or SDTP)
        """
        self.dataset = LongBenchDataset(task_path)
        self.model = model
        self.task_path = task_path

    def evaluate(self, num_samples: Optional[int] = None) -> Dict:
        """
        Only print evaluation plan — no inference performed.
        
        Args:
            num_samples: Number of samples to evaluate (None = all)
            
        Returns:
            Dict with evaluation setup information
        """
        total_samples = len(self.dataset)
        eval_samples = num_samples if num_samples is not None else total_samples
        
        print(f"[Eval] Task loaded: {total_samples} total samples")
        print(f"[Eval] Will evaluate: {eval_samples} samples")
        print(f"[Eval] Model prepared: {self.model.model_name}")
        print(f"[Eval] Mode: {'SDTP' if self.model.is_sdtp() else 'Baseline'}")
        print(f"[Eval] No inference is executed at this stage.")
        
        return {
            "task_path": self.task_path,
            "task_size": total_samples,
            "eval_samples": eval_samples,
            "model": self.model.model_name,
            "mode": "SDTP" if self.model.is_sdtp() else "Baseline",
            "pruning_module": self.model.pruning_module_path,
            "status": "setup_completed",
        }
    
    def get_sample(self, index: int) -> Dict:
        """
        Get a sample from the dataset (no inference).
        
        Args:
            index: Sample index
            
        Returns:
            Sample dict with 'input' and 'answers'
        """
        return self.dataset[index]
    
    def get_all_samples(self) -> List[Dict]:
        """Get all samples from the dataset"""
        return self.dataset.get_all_samples()

