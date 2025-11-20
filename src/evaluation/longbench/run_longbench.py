import argparse
import json
import os
from .model_wrapper import ModelWrapper
from .evaluator import LongBenchEvaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model", type=str, default="checkpoints/qwen2-7b-instruct")
    parser.add_argument("--pruning_module", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/longbench_result.json")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "sdtp"])
    parser.add_argument("--do_inference", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--prediction_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    # ****** NEW ARGUMENT FOR SDTP ******
    parser.add_argument("--keep_ratio", type=float, default=1.0,
                        help="Token keep ratio for SDTP pruning (default=1.0 = no pruning)")

    args = parser.parse_args()

    print("[LongBench] Starting evaluation...")
    print(f"  Task: {args.task}")
    print(f"  Model: {args.model}")

    model = ModelWrapper(
        model_name=args.model,
        pruning_module_path=args.pruning_module,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        keep_ratio=args.keep_ratio,      # *** NEW LINE ***
    )

    model.load_model(real_load=args.do_inference)

    evaluator = LongBenchEvaluator(
        task_path=args.task,
        model=model,
        max_samples=args.max_samples,
    )

    result = evaluator.evaluate(
        do_inference=args.do_inference,
        save_predictions=args.save_predictions,
        prediction_path=args.prediction_path,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] Summary saved to {args.output}")

if __name__ == "__main__":
    main()
