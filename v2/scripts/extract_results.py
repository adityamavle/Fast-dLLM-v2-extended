#!/usr/bin/env python3
"""
Extract and format results from sweep JSONL for report generation
"""
import argparse
import json
from pathlib import Path
from tabulate import tabulate

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

def format_table(results, skip_type):
    """Format results as a table"""
    if skip_type == "token":
        filtered = [r for r in results if r.get('token_skip') and not r.get('layer_skip')]
        tau_key = 'token_tau'
    elif skip_type == "layer":
        filtered = [r for r in results if r.get('layer_skip') and not r.get('token_skip')]
        tau_key = 'layer_tau'
    else:
        return None
    
    # Sort by threshold
    filtered.sort(key=lambda x: x.get(tau_key, 0))
    
    table_data = []
    for r in filtered:
        table_data.append([
            f"{r.get(tau_key, 0):.3f}",
            f"{r.get('accuracy', 0):.4f}" if r.get('accuracy') is not None else "N/A",
            f"{r.get('flops_reduction', 0)*100:.2f}%" if r.get('flops_reduction') is not None else "N/A"
        ])
    
    return table_data

def main():
    parser = argparse.ArgumentParser(description="Extract results for report")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", help="Output markdown file (optional)")
    args = parser.parse_args()
    
    results = load_results(args.input)
    if not results:
        print(f"Error: No results found in {args.input}")
        return
    
    # Find baseline
    baseline = [r for r in results if not r.get('token_skip') and not r.get('layer_skip')]
    baseline_acc = baseline[0].get('accuracy', 0.0) if baseline else None
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nBaseline Accuracy: {baseline_acc:.4f}" if baseline_acc else "\nBaseline: Not found")
    
    # Token skipping table
    print("\n" + "-"*60)
    print("Token-Level Skipping Results")
    print("-"*60)
    token_table = format_table(results, "token")
    if token_table:
        print(tabulate(token_table, headers=["Threshold (τ)", "Accuracy", "FLOPs Reduction"], tablefmt="grid"))
    
    # Layer skipping table
    print("\n" + "-"*60)
    print("Layer-Level Skipping Results")
    print("-"*60)
    layer_table = format_table(results, "layer")
    if layer_table:
        print(tabulate(layer_table, headers=["Threshold (τ)", "Accuracy", "FLOPs Reduction"], tablefmt="grid"))
    
    # Summary statistics
    print("\n" + "-"*60)
    print("Summary Statistics")
    print("-"*60)
    
    if token_table:
        token_reductions = [float(r.get('flops_reduction', 0)*100) for r in results if r.get('token_skip') and r.get('flops_reduction') is not None]
        if token_reductions:
            print(f"Token Skipping - Max FLOPs Reduction: {max(token_reductions):.2f}%")
            print(f"Token Skipping - Min FLOPs Reduction: {min(token_reductions):.2f}%")
    
    if layer_table:
        layer_reductions = [float(r.get('flops_reduction', 0)*100) for r in results if r.get('layer_skip') and r.get('flops_reduction') is not None]
        if layer_reductions:
            print(f"Layer Skipping - Max FLOPs Reduction: {max(layer_reductions):.2f}%")
            print(f"Layer Skipping - Min FLOPs Reduction: {min(layer_reductions):.2f}%")
    
    # Generate markdown if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("# Results Summary\n\n")
            f.write(f"Baseline Accuracy: {baseline_acc:.4f}\n\n" if baseline_acc else "Baseline: Not found\n\n")
            
            f.write("## Token-Level Skipping\n\n")
            if token_table:
                f.write(tabulate(token_table, headers=["Threshold (τ)", "Accuracy", "FLOPs Reduction"], tablefmt="pipe"))
                f.write("\n\n")
            
            f.write("## Layer-Level Skipping\n\n")
            if layer_table:
                f.write(tabulate(layer_table, headers=["Threshold (τ)", "Accuracy", "FLOPs Reduction"], tablefmt="pipe"))
                f.write("\n\n")
        
        print(f"\nMarkdown summary saved to {args.output}")

if __name__ == "__main__":
    main()
