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
    
    # Separate results by method (using 'method' field if available, fall back to token_skip/layer_skip)
    baseline = [r for r in results if r.get('method', '').lower() == 'baseline' or 
                (not r.get('token_skip', False) and not r.get('layer_skip', False))]
    token_only = [r for r in results if r.get('method', '').lower() == 'token_skip' or 
                  (r.get('token_skip', False) and not r.get('layer_skip', False))]
    layer_only = [r for r in results if r.get('method', '').lower() == 'layer_skip' or 
                  (not r.get('token_skip', False) and r.get('layer_skip', False))]
    combined = [r for r in results if r.get('method', '').lower() == 'combined' or 
                (r.get('token_skip', False) and r.get('layer_skip', False))]
    
    plt.figure(figsize=(11, 8))
    
    # Extract baseline accuracy
    baseline_acc = None
    if baseline:
        baseline_acc = baseline[0].get('accuracy', 0.0)
        if baseline_acc is not None and isinstance(baseline_acc, (int, float)):
            plt.axhline(y=baseline_acc, color='black', linestyle='--', 
                       label=f'Baseline (acc={baseline_acc:.4f})', linewidth=2.5)
    
    # Plot token-only skipping
    if token_only:
        # Filter valid pairs and sort by FLOPs reduction
        token_data = []
        for r in token_only:
            flops = r.get('flops_reduction')
            acc = r.get('accuracy')
            if flops is not None and acc is not None and isinstance(flops, (int, float)) and isinstance(acc, (int, float)):
                token_data.append((flops, acc, r.get('token_tau', 0)))
        
        if token_data:
            token_data.sort(key=lambda x: x[0])  # Sort by FLOPs reduction
            token_flops, token_acc, token_taus = zip(*token_data)
            plt.plot(token_flops, token_acc, 'o-', label='Token-level skipping', 
                    linewidth=2.5, markersize=10, color='blue', markeredgewidth=1.5, markeredgecolor='darkblue')
            
            # Add tau annotations
            for flops, acc, tau in token_data:
                plt.annotate(f'τ={tau:.2f}', 
                            (flops, acc),
                            fontsize=9, alpha=0.8, xytext=(8, 8), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
    
    # Plot layer-only skipping
    if layer_only:
        # Filter valid pairs and sort by FLOPs reduction
        layer_data = []
        for r in layer_only:
            flops = r.get('flops_reduction')
            acc = r.get('accuracy')
            if flops is not None and acc is not None and isinstance(flops, (int, float)) and isinstance(acc, (int, float)):
                layer_data.append((flops, acc, r.get('layer_tau', 0)))
        
        if layer_data:
            layer_data.sort(key=lambda x: x[0])  # Sort by FLOPs reduction
            layer_flops, layer_acc, layer_taus = zip(*layer_data)
            plt.plot(layer_flops, layer_acc, 's-', label='Layer-level skipping',
                    linewidth=2.5, markersize=10, color='red', markeredgewidth=1.5, markeredgecolor='darkred')
            
            # Add tau annotations
            for flops, acc, tau in layer_data:
                plt.annotate(f'τ={tau:.2f}', 
                            (flops, acc),
                            fontsize=9, alpha=0.8, xytext=(8, -15), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.5))
    
    # Plot combined skipping
    if combined:
        # Filter valid pairs and sort by FLOPs reduction
        comb_data = []
        for r in combined:
            flops = r.get('flops_reduction')
            acc = r.get('accuracy')
            if flops is not None and acc is not None and isinstance(flops, (int, float)) and isinstance(acc, (int, float)):
                comb_data.append((flops, acc, r.get('token_tau', 0), r.get('layer_tau', 0)))
        
        if comb_data:
            comb_data.sort(key=lambda x: x[0])  # Sort by FLOPs reduction
            comb_flops, comb_acc, _, _ = zip(*comb_data)
            plt.plot(comb_flops, comb_acc, '^-', label='Combined skipping',
                    linewidth=2.5, markersize=10, color='green', markeredgewidth=1.5, markeredgecolor='darkgreen')
    
    plt.xlabel('FLOPs Reduction', fontsize=13, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
    plt.title(f'Accuracy vs FLOPs Reduction: {task_name}', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    plt.legend(fontsize=11, loc='best', framealpha=0.95)
    plt.xlim(left=-0.02, right=1.02)
    
    # Set y-axis limits based on data with padding
    all_acc = [r.get('accuracy', 0.0) for r in results if r.get('accuracy') is not None and isinstance(r.get('accuracy'), (int, float))]
    if all_acc:
        y_min = max(0, min(all_acc) - 0.10)
        y_max = min(1.0, max(all_acc) + 0.10)
        plt.ylim(bottom=y_min, top=y_max)
    else:
        plt.ylim(bottom=0, top=1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot accuracy vs FLOPs reduction curves")
    parser.add_argument("--input", required=True, help="Input JSONL file with results")
    parser.add_argument("--output", required=True, help="Output PNG file")
    parser.add_argument("--task", default="", help="Task name for plot title")
    args = parser.parse_args()
    
    results = load_results(args.input)
    if not results:
        print(f"✗ Error: No valid results found in {args.input}")
        return
    
    print(f"✓ Loaded {len(results)} result entries from {args.input}")
    
    # Validate data integrity
    valid_results = []
    for r in results:
        if 'accuracy' in r and 'flops_reduction' in r:
            valid_results.append(r)
    
    if len(valid_results) != len(results):
        print(f"⚠ Warning: {len(results) - len(valid_results)} entries missing required fields, using {len(valid_results)} valid entries")
    
    if not valid_results:
        print(f"✗ Error: No results with both 'accuracy' and 'flops_reduction' fields")
        return
    
    task_name = args.task or valid_results[0].get('task', 'Evaluation')
    print(f"✓ Generating plot: {task_name}")
    plot_accuracy_vs_flops(valid_results, args.output, task_name)

if __name__ == "__main__":
    main()

