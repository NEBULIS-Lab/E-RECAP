"""
Parse latency log files and convert to JSON format
Supports parsing from inference output logs
"""
import re
import json
import argparse
import os


def parse_log_file(log_path, baseline_out, sdtp_out):
    """
    Parse log file with format:
    [Length 4096] baseline=0.7065s  sdtp=0.2527s  speedup=2.80x
    
    Args:
        log_path: Path to log file
        baseline_out: Output path for baseline JSON
        sdtp_out: Output path for SDTP JSON
    """
    # Pattern to match: [Length 4096] baseline=0.7065s  sdtp=0.2527s  speedup=2.80x
    pattern = re.compile(
        r"\[Length\s+(\d+)\].*baseline=([\d.]+)s.*sdtp=([\d.]+)s"
    )
    
    baseline = {}
    sdtp = {}
    
    if not os.path.exists(log_path):
        print(f"[Error] Log file not found: {log_path}")
        return
    
    print(f"[Parsing] Reading log file: {log_path}")
    
    with open(log_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            match = pattern.search(line)
            if match:
                length = int(match.group(1))
                baseline_latency = float(match.group(2))
                sdtp_latency = float(match.group(3))
                
                baseline[length] = baseline_latency
                sdtp[length] = sdtp_latency
                
                print(f"  Found: Length={length}, baseline={baseline_latency:.4f}s, "
                      f"sdtp={sdtp_latency:.4f}s")
    
    if not baseline:
        print("[Warning] No matching patterns found in log file")
        print("[Info] Expected format: [Length 4096] baseline=0.7065s  sdtp=0.2527s  speedup=2.80x")
        return
    
    # Save as JSON with string keys (will be converted to int when loading)
    os.makedirs(os.path.dirname(baseline_out), exist_ok=True)
    with open(baseline_out, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    os.makedirs(os.path.dirname(sdtp_out), exist_ok=True)
    with open(sdtp_out, 'w') as f:
        json.dump(sdtp, f, indent=2)
    
    print(f"[OK] Parsed {len(baseline)} data points")
    print(f"[OK] Baseline data saved to: {baseline_out}")
    print(f"[OK] SDTP data saved to: {sdtp_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse latency log files and convert to JSON"
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to log file to parse"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="results/latency_baseline.json",
        help="Output path for baseline JSON"
    )
    parser.add_argument(
        "--sdtp",
        type=str,
        default="results/latency_sdtp.json",
        help="Output path for SDTP JSON"
    )
    
    args = parser.parse_args()
    
    parse_log_file(args.log, args.baseline, args.sdtp)


if __name__ == "__main__":
    main()

