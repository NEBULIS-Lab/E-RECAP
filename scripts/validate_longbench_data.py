#!/usr/bin/env python3
"""
Validate LongBench v1 dataset files to ensure they match E-RECAP format.
"""

import json
import sys
from pathlib import Path
from typing import Tuple


def validate_json_file(file_path: Path) -> Tuple[bool, str]:
    """
    Validate a LongBench JSON file.
    
    Returns:
        (is_valid, message)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Must be a list
        if not isinstance(data, list):
            return False, f"File is not a list (got {type(data).__name__})"
        
        if len(data) == 0:
            return False, "File is empty"
        
        # Validate each item
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                return False, f"Item {i} is not a dict (got {type(item).__name__})"
            
            if "input" not in item:
                return False, f"Item {i} missing 'input' field"
            
            if "answers" not in item:
                return False, f"Item {i} missing 'answers' field"
            
            if not isinstance(item["answers"], list):
                return False, f"Item {i}: 'answers' is not a list (got {type(item['answers']).__name__})"
        
        return True, f"Valid: {len(data)} items, format correct"
        
    except FileNotFoundError:
        return False, "File not found"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Validate all LongBench dataset files."""
    data_dir = Path(__file__).parent.parent / "data" / "LongBench_data"
    
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("LongBench v1 Dataset Validation")
    print("=" * 60)
    print(f"Data directory: {data_dir}\n")
    
    json_files = sorted(data_dir.glob("*.json"))
    
    if not json_files:
        print("No JSON files found in data directory")
        sys.exit(1)
    
    all_valid = True
    
    for json_file in json_files:
        is_valid, message = validate_json_file(json_file)
        status = "✓" if is_valid else "✗"
        print(f"{status} {json_file.name}: {message}")
        if not is_valid:
            all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✓ All files are valid!")
        sys.exit(0)
    else:
        print("✗ Some files are invalid")
        sys.exit(1)


if __name__ == "__main__":
    main()
