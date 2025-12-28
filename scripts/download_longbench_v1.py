#!/usr/bin/env python3
"""
Download and prepare LongBench v1 datasets for E-RECAP project.
Converts datasets to unified JSON format with 'input' and 'answers' fields.
"""

import json
import os
import sys
from pathlib import Path
from datasets import load_dataset
from typing import List, Dict, Any


def convert_dataset_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert LongBench dataset item to E-RECAP format.
    
    Expected output:
    {
        "input": "...",
        "answers": ["..."]
    }
    """
    # Extract input (usually 'input' field)
    input_text = item.get("input", "")
    
    # Extract answers (usually 'answers' field, ensure it's a list)
    answers = item.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    elif not isinstance(answers, list):
        answers = [str(answers)] if answers else [""]
    
    # Ensure answers is not empty
    if not answers:
        answers = [""]
    
    return {
        "input": str(input_text),
        "answers": [str(ans) for ans in answers]
    }


def download_and_save_dataset(
    dataset_name: str,
    output_path: Path,
    split: str = "test"
) -> bool:
    """
    Download LongBench v1 dataset and save in E-RECAP format.
    
    Args:
        dataset_name: Name of the dataset (e.g., "narrativeqa")
        output_path: Path to save the JSON file
        split: Dataset split to use ("test", "validation", or "train")
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n[{dataset_name}] Downloading from Hugging Face...")
    
    try:
        # Try test split first
        try:
            dataset = load_dataset(
                "THUDM/LongBench", 
                dataset_name, 
                split=split,
                trust_remote_code=True
            )
            print(f"  ✓ Loaded split: {split}")
        except Exception as e:
            print(f"  ⚠ Split '{split}' not available: {e}")
            # Try validation or train
            for alt_split in ["validation", "train", "test"]:
                if alt_split == split:
                    continue
                try:
                    dataset = load_dataset(
                        "THUDM/LongBench", 
                        dataset_name, 
                        split=alt_split,
                        trust_remote_code=True
                    )
                    print(f"  ✓ Loaded alternative split: {alt_split}")
                    break
                except:
                    continue
            else:
                raise Exception(f"Could not load dataset with any split")
        
        # Convert to E-RECAP format
        print(f"  Converting {len(dataset)} items...")
        converted_data = []
        
        for i, item in enumerate(dataset):
            try:
                converted_item = convert_dataset_item(item)
                converted_data.append(converted_item)
                
                if (i + 1) % 100 == 0:
                    print(f"    Processed {i + 1}/{len(dataset)} items...")
            except Exception as e:
                print(f"    ⚠ Warning: Failed to convert item {i}: {e}")
                continue
        
        # Save as JSON (UTF-8, no BOM)
        print(f"  Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        # Validate saved file
        with open(output_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
            
        if not isinstance(saved_data, list):
            print(f"  ✗ Error: Saved file is not a list")
            return False
        
        if len(saved_data) == 0:
            print(f"  ⚠ Warning: Saved file is empty")
            return False
        
        # Validate structure
        first_item = saved_data[0]
        if "input" not in first_item or "answers" not in first_item:
            print(f"  ✗ Error: Invalid structure in saved file")
            return False
        
        if not isinstance(first_item["answers"], list):
            print(f"  ✗ Error: 'answers' is not a list")
            return False
        
        print(f"  ✓ Successfully saved {len(saved_data)} items")
        print(f"  ✓ Validation passed: format is correct")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to download all LongBench v1 datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download LongBench v1 datasets")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files without prompting"
    )
    args = parser.parse_args()
    
    # Output directory
    output_dir = Path(__file__).parent.parent / "data" / "LongBench_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LongBench v1 Dataset Downloader for E-RECAP")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()
    
    # List of datasets to download
    datasets = [
        "narrativeqa",
        "qasper",
        "gov_report",
        "multi_news",
        "multifieldqa_en",
        "hotpotqa",
        "musique",
        "triviaqa",
        # Try legal_contract_qa (might not exist in LongBench v1)
        "legal_contract_qa",
    ]
    
    results = {}
    
    for dataset_name in datasets:
        output_path = output_dir / f"{dataset_name}.json"
        
        # Skip if already exists (unless overwrite is set)
        if output_path.exists() and not args.overwrite:
            print(f"\n[{dataset_name}] File already exists: {output_path}")
            print(f"  Skipped (use --overwrite to replace)")
            results[dataset_name] = "skipped"
            continue
        
        success = download_and_save_dataset(dataset_name, output_path)
        results[dataset_name] = "success" if success else "failed"
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    for dataset_name, status in results.items():
        status_symbol = {
            "success": "✓",
            "failed": "✗",
            "skipped": "⊘"
        }.get(status, "?")
        print(f"  {status_symbol} {dataset_name}: {status}")
    
    successful = sum(1 for s in results.values() if s == "success")
    total = len(results)
    
    print(f"\nCompleted: {successful}/{total} datasets downloaded successfully")
    print(f"Output directory: {output_dir}")
    
    if successful < total:
        print("\n⚠ Some datasets failed to download. Check errors above.")
        sys.exit(1)
    else:
        print("\n✓ All datasets downloaded successfully!")


if __name__ == "__main__":
    main()
