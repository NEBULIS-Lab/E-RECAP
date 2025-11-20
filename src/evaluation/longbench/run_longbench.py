"""
LongBench evaluation main script (setup phase - no inference)
Main entry point for LongBench evaluation framework
"""
import argparse
import json
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.evaluation.longbench.model_wrapper import ModelWrapper
from src.evaluation.longbench.evaluator import LongBenchEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="LongBench evaluation setup (no inference execution)"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Path to LongBench task JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/qwen2-7b-instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--pruning_module",
        type=str,
        default=None,
        help="Path to pruning module checkpoint (None for baseline)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/longbench_result.json",
        help="Output JSON path for setup result"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)"
    )
    
    args = parser.parse_args()
    
    # Check if task file exists
    if not os.path.exists(args.task):
        print(f"[Error] Task file not found: {args.task}")
        print("[Info] LongBench task files should be JSON format.")
        return
    
    # Initialize model wrapper (no actual loading)
    model = ModelWrapper(args.model, pruning_module_path=args.pruning_module)
    model.load_model()  # Does not actually load
    
    # Initialize evaluator
    evaluator = LongBenchEvaluator(args.task, model)
    
    # Run evaluation setup (no inference)
    result = evaluator.evaluate(num_samples=args.num_samples)
    
    # Save setup result
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Written setup result to {args.output}")


if __name__ == "__main__":
    main()

