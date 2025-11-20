"""
Latency curve plotting script for SDTP
Generates latency, speedup, and FLOPs reduction curves
"""
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


def load_json(path):
    """Load JSON file and convert keys to integers"""
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return {}
    
    with open(path, 'r') as f:
        data = json.load(f)
        return {int(k): float(v) for k, v in data.items()}


def plot_latency(baseline, sdtp, out_path):
    """
    Plot prefill latency vs sequence length
    
    Args:
        baseline: Dict mapping sequence length to latency (seconds)
        sdtp: Dict mapping sequence length to latency (seconds)
        out_path: Output file path
    """
    # Get common lengths
    lengths = sorted(set(baseline.keys()) & set(sdtp.keys()))
    if not lengths:
        print("[Error] No common sequence lengths found")
        return
    
    base_vals = [baseline[L] for L in lengths]
    sdtp_vals = [sdtp[L] for L in lengths]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, base_vals, marker='o', linewidth=2, label="Baseline", markersize=8)
    plt.plot(lengths, sdtp_vals, marker='s', linewidth=2, label="SDTP", markersize=8)
    plt.xlabel("Sequence Length (tokens)", fontsize=12)
    plt.ylabel("Prefill Latency (seconds)", fontsize=12)
    plt.title("Prefill Latency vs Sequence Length", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved {out_path}")


def plot_speedup(baseline, sdtp, out_path):
    """
    Plot speedup vs sequence length
    
    Args:
        baseline: Dict mapping sequence length to latency (seconds)
        sdtp: Dict mapping sequence length to latency (seconds)
        out_path: Output file path
    """
    # Get common lengths
    lengths = sorted(set(baseline.keys()) & set(sdtp.keys()))
    if not lengths:
        print("[Error] No common sequence lengths found")
        return
    
    speedups = [baseline[L] / sdtp[L] if sdtp[L] > 0 else 0 for L in lengths]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, speedups, marker='o', linewidth=2, label="Speedup", 
             markersize=8, color='green')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline (1x)')
    plt.xlabel("Sequence Length (tokens)", fontsize=12)
    plt.ylabel("Speedup (Baseline / SDTP)", fontsize=12)
    plt.title("Speedup vs Sequence Length", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved {out_path}")
    print(f"[Info] Average speedup: {np.mean(speedups):.2f}x")


def estimate_flops(length, keep_ratio=0.7, hidden_size=3584, num_layers=28, num_heads=32):
    """
    Estimate FLOPs for Transformer forward pass
    
    Args:
        length: Sequence length
        keep_ratio: Token keep ratio (for SDTP)
        hidden_size: Model hidden size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Attention head dimension
    
    Returns:
        Estimated FLOPs (normalized)
    """
    head_dim = hidden_size // num_heads
    
    # Attention FLOPs: 4 * L^2 * d (QK^T, softmax, AV)
    # MLP FLOPs: 8 * L * d^2 (two linear layers, expansion ratio ~4)
    # Per layer FLOPs
    attn_flops = 4 * length * length * hidden_size
    mlp_flops = 8 * length * hidden_size * hidden_size
    
    # Total FLOPs for all layers
    total_flops = num_layers * (attn_flops + mlp_flops)
    
    return total_flops


def plot_flops(baseline, sdtp, out_path, keep_ratio=0.7):
    """
    Plot estimated FLOPs reduction
    
    Args:
        baseline: Dict mapping sequence length to latency (for reference)
        sdtp: Dict mapping sequence length to latency (for reference)
        out_path: Output file path
        keep_ratio: Average token keep ratio for SDTP
    """
    # Get common lengths
    lengths = sorted(set(baseline.keys()) & set(sdtp.keys()))
    if not lengths:
        print("[Error] No common sequence lengths found")
        return
    
    # Estimate FLOPs
    base_flops = [estimate_flops(L) for L in lengths]
    
    # SDTP FLOPs: tokens are pruned progressively, use average keep ratio
    # For simplicity, use keep_ratio for all layers (in reality it's progressive)
    sdtp_flops = [estimate_flops(int(L * keep_ratio)) for L in lengths]
    
    # Normalize to first value for better visualization
    if base_flops[0] > 0:
        base_flops_norm = [f / base_flops[0] for f in base_flops]
        sdtp_flops_norm = [f / base_flops[0] for f in sdtp_flops]
    else:
        base_flops_norm = base_flops
        sdtp_flops_norm = sdtp_flops
    
    reduction = [(1 - sdtp_flops[i] / base_flops[i]) * 100 
                 for i in range(len(lengths))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, base_flops_norm, marker='o', linewidth=2, 
             label="Baseline FLOPs", markersize=8)
    plt.plot(lengths, sdtp_flops_norm, marker='s', linewidth=2, 
             label=f"SDTP FLOPs (keep_ratio={keep_ratio})", markersize=8)
    plt.xlabel("Sequence Length (tokens)", fontsize=12)
    plt.ylabel("Relative FLOPs (normalized)", fontsize=12)
    plt.title("Estimated FLOPs Reduction", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved {out_path}")
    print(f"[Info] Average FLOPs reduction: {np.mean(reduction):.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Generate latency curves for SDTP evaluation"
    )
    parser.add_argument(
        "--baseline", 
        type=str, 
        default="results/latency_baseline.json",
        help="Path to baseline latency JSON file"
    )
    parser.add_argument(
        "--sdtp", 
        type=str, 
        default="results/latency_sdtp.json",
        help="Path to SDTP latency JSON file"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="results/fig",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.7,
        help="Token keep ratio for FLOPs estimation"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output filenames (e.g., 'singlegpu_' or 'multigpu_')"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"[Loading] Baseline: {args.baseline}")
    baseline = load_json(args.baseline)
    
    print(f"[Loading] SDTP: {args.sdtp}")
    sdtp = load_json(args.sdtp)
    
    if not baseline:
        print("[Error] Baseline data is empty")
        return
    
    if not sdtp:
        print("[Error] SDTP data is empty")
        return
    
    print(f"[Info] Found {len(baseline)} baseline points, {len(sdtp)} SDTP points")
    
    # Generate plots with optional prefix
    prefix = f"{args.prefix}_" if args.prefix else ""
    plot_latency(baseline, sdtp, os.path.join(args.out_dir, f"{prefix}latency_curve.png"))
    plot_speedup(baseline, sdtp, os.path.join(args.out_dir, f"{prefix}speedup_curve.png"))
    plot_flops(baseline, sdtp, os.path.join(args.out_dir, f"{prefix}flops_curve.png"), 
               keep_ratio=args.keep_ratio)
    
    print("[OK] All plots generated successfully!")


if __name__ == "__main__":
    main()

