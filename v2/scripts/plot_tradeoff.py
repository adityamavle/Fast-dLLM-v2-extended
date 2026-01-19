#!/usr/bin/env python3
"""
Plot accuracy vs FLOPs reduction curves
"""
import argparse
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(jsonl_path):
    """Load results from JSONL file"""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results

def plot_accuracy_vs_flops(results, output_path, task_name="Task"):
    """Plot accuracy vs FLOPs reduction with different series"""
    
    # Separate results by skip type
    baseline = [r for r in results if not r.get('token_skip', False) and not r.get('layer_skip', False)]
    token_only = [r for r in results if r.get('token_skip', False) and not r.get('layer_skip', False)]
    layer_only = [r for r in results if not r.get('token_skip', False) and r.get('layer_skip', False)]
    combined = [r for r in results if r.get('token_skip', False) and r.get('layer_skip', False)]
    
    plt.figure(figsize=(10, 7))
    
    # Plot baseline
    if baseline:
        baseline_acc = baseline[0].get('accuracy', 0.0)
        if baseline_acc is not None:
            plt.axhline(y=baseline_acc, color='black', linestyle='--', 
                       label=f'Baseline (acc={baseline_acc:.3f})', linewidth=2)
    
    # Plot token-only skipping
    if token_only:
        token_flops = [r.get('flops_reduction', 0.0) for r in token_only if r.get('flops_reduction') is not None]
        token_acc = [r.get('accuracy', 0.0) for r in token_only if r.get('flops_reduction') is not None]
        # Filter out None values
        valid_pairs = [(f, a) for f, a in zip(token_flops, token_acc) if f is not None and a is not None]
        if valid_pairs:
            token_flops, token_acc = zip(*sorted(valid_pairs))
            plt.plot(token_flops, token_acc, 'o-', label='Token-level skipping', 
                    linewidth=2, markersize=8, color='blue')
    
    # Plot layer-only skipping
    if layer_only:
        layer_flops = [r.get('flops_reduction', 0.0) for r in layer_only if r.get('flops_reduction') is not None]
        layer_acc = [r.get('accuracy', 0.0) for r in layer_only if r.get('flops_reduction') is not None]
        valid_pairs = [(f, a) for f, a in zip(layer_flops, layer_acc) if f is not None and a is not None]
        if valid_pairs:
            layer_flops, layer_acc = zip(*sorted(valid_pairs))
            plt.plot(layer_flops, layer_acc, 's-', label='Layer-level skipping',
                    linewidth=2, markersize=8, color='red')
    
    # Plot combined
    if combined:
        comb_flops = [r.get('flops_reduction', 0.0) for r in combined if r.get('flops_reduction') is not None]
        comb_acc = [r.get('accuracy', 0.0) for r in combined if r.get('flops_reduction') is not None]
        valid_pairs = [(f, a) for f, a in zip(comb_flops, comb_acc) if f is not None and a is not None]
        if valid_pairs:
            comb_flops, comb_acc = zip(*sorted(valid_pairs))
            plt.plot(comb_flops, comb_acc, '^-', label='Combined skipping',
                    linewidth=2, markersize=8, color='green')
    
    plt.xlabel('FLOPs Reduction', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Accuracy vs FLOPs Reduction: {task_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='best')
    plt.xlim(left=0)
    
    # Set y-axis limits based on data
    all_acc = [r.get('accuracy', 0.0) for r in results if r.get('accuracy') is not None]
    if all_acc:
        y_min = max(0, min(all_acc) - 0.05)
        y_max = min(1.0, max(all_acc) + 0.05)
        plt.ylim(bottom=y_min, top=y_max)
    else:
        plt.ylim(bottom=0, top=1.0)
    
    # Add text annotations for thresholds on token-only line
    if token_only:
        for r in token_only:
            if r.get('token_tau') and r.get('flops_reduction') is not None and r.get('accuracy') is not None:
                plt.annotate(f"τ={r['token_tau']:.3f}", 
                            (r.get('flops_reduction', 0), r.get('accuracy', 0)),
                            fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot accuracy vs FLOPs reduction curves")
    parser.add_argument("--input", required=True, help="Input JSONL file with results")
    parser.add_argument("--output", required=True, help="Output PNG file")
    parser.add_argument("--task", default="", help="Task name for plot title")
    args = parser.parse_args()
    
    results = load_results(args.input)
    if not results:
        print(f"Error: No results found in {args.input}")
        return
    
    task_name = args.task or results[0].get('task', 'Task')
    plot_accuracy_vs_flops(results, args.output, task_name)

if __name__ == "__main__":
    main()

