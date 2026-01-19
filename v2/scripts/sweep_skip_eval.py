#!/usr/bin/env python3
"""
Sweep script to run evaluations with different skip thresholds
and collect accuracy vs FLOPs reduction data
"""
import argparse
import subprocess
import json
import time
from pathlib import Path
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def parse_lm_eval_output(output_dir, task_name):
    """Parse lm_eval JSON output to extract accuracy"""
    # lm_eval saves results in output_dir/results.json
    results_file = Path(output_dir) / "results.json"
    if not results_file.exists():
        return None
    
    try:
        with open(results_file) as f:
            results = json.load(f)
        
        # Extract task-specific metric (usually 'acc' or 'exact_match')
        task_results = results.get('results', {}).get(task_name, {})
        
        # Try common metric names
        for metric in ['acc', 'exact_match', 'acc_norm', 'acc_stderr']:
            if metric in task_results:
                return task_results[metric]
        
        # If no standard metric, return first numeric value
        for key, value in task_results.items():
            if isinstance(value, (int, float)):
                return value
    except Exception as e:
        print(f"Error parsing results: {e}")
    
    return None

def run_evaluation(task, token_tau, layer_tau, token_skip, layer_skip, 
                   output_base_dir, stats_base_dir):
    """Run a single evaluation with given parameters"""
    
    # Create unique output directory
    param_str = f"t{token_tau}_l{layer_tau}_ts{token_skip}_ls{layer_skip}"
    output_dir = Path(output_base_dir) / f"{task}_{param_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats_file = Path(stats_base_dir) / f"{task}_{param_str}.jsonl"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Build model_args string
    model_args = (
        f"model_path=Efficient-Large-Model/Fast_dLLM_v2_7B,"
        f"token_skip={token_skip},"
        f"token_tau={token_tau},"
        f"layer_skip={layer_skip},"
        f"layer_tau={layer_tau},"
        f"skip_stats_path={stats_file},"
        f"threshold=0.9"
    )
    
    # Build command
    cmd = [
        "accelerate", "launch", "eval.py",
        "--tasks", task,
        "--batch_size", "32",
        "--num_fewshot", "0",
        "--confirm_run_unsafe_code",
        "--model", "fast_dllm_v2",
        "--fewshot_as_multiturn",
        "--apply_chat_template",
        "--model_args", model_args,
        "--output_path", str(output_dir),
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {task} | token_tau={token_tau}, layer_tau={layer_tau}")
    print(f"token_skip={token_skip}, layer_skip={layer_skip}")
    print(f"{'='*60}\n")
    
    # Run evaluation
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"ERROR: Evaluation failed")
        print(result.stderr)
        return None
    
    # Parse accuracy
    accuracy = parse_lm_eval_output(output_dir, task)
    
    # Load FLOPs stats
    flops_reduction = None
    if stats_file.exists():
        try:
            with open(stats_file) as f:
                for line in f:
                    if line.strip():
                        stats_data = json.loads(line)
                        flops_reduction = stats_data.get('flops_reduction', None)
                        break
        except Exception as e:
            print(f"Error loading stats: {e}")
    
    return {
        'task': task,
        'token_tau': token_tau,
        'layer_tau': layer_tau,
        'token_skip': token_skip,
        'layer_skip': layer_skip,
        'accuracy': accuracy,
        'flops_reduction': flops_reduction,
        'output_dir': str(output_dir),
    }

def main():
    parser = argparse.ArgumentParser(description="Sweep skip thresholds and collect results")
    parser.add_argument("--task", default="gsm8k", help="Evaluation task (gsm8k, mmlu, etc.)")
    parser.add_argument("--output_dir", default="./results/eval_outputs", help="Output directory for lm_eval results")
    parser.add_argument("--stats_dir", default="./results/flops_stats", help="Directory for FLOPs stats")
    parser.add_argument("--results_file", default="./results/sweep_results.jsonl", help="Combined results JSONL")
    parser.add_argument("--token_taus", nargs="+", type=float, 
                       default=[0.95, 0.97, 0.99, 0.995, 0.999],
                       help="Token skip thresholds to sweep")
    parser.add_argument("--layer_taus", nargs="+", type=float,
                       default=[0.95, 0.97, 0.99, 0.995, 0.999],
                       help="Layer skip thresholds to sweep")
    parser.add_argument("--baseline_only", action="store_true", 
                       help="Only run baseline (no skipping)")
    args = parser.parse_args()
    
    results = []
    
    # Baseline (no skipping)
    print("Running baseline (no skipping)...")
    baseline = run_evaluation(
        args.task, 0.0, 0.0, False, False,
        args.output_dir, args.stats_dir
    )
    if baseline:
        baseline['accuracy'] = baseline.get('accuracy', 0.0)
        baseline['flops_reduction'] = 0.0
        results.append(baseline)
        print(f"Baseline: accuracy={baseline['accuracy']}, flops_reduction=0.0")
    
    if args.baseline_only:
        # Save results
        results_file = Path(args.results_file)
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        return
    
    # Sweep token skipping only
    print("\nSweeping token skipping...")
    for tau in args.token_taus:
        result = run_evaluation(
            args.task, tau, 0.0, True, False,
            args.output_dir, args.stats_dir
        )
        if result:
            results.append(result)
            print(f"Token tau={tau}: accuracy={result.get('accuracy')}, "
                  f"flops_reduction={result.get('flops_reduction')}")
        time.sleep(2)  # Brief pause between runs
    
    # Sweep layer skipping only
    print("\nSweeping layer skipping...")
    for tau in args.layer_taus:
        result = run_evaluation(
            args.task, 0.0, tau, False, True,
            args.output_dir, args.stats_dir
        )
        if result:
            results.append(result)
            print(f"Layer tau={tau}: accuracy={result.get('accuracy')}, "
                  f"flops_reduction={result.get('flops_reduction')}")
        time.sleep(2)
    
    # Save combined results
    results_file = Path(args.results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\n{'='*60}")
    print(f"Saved {len(results)} results to {results_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

