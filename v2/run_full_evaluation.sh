#!/bin/bash
# run_full_evaluation.sh
# Complete evaluation script for token and layer skipping

set -e  # Exit on error

cd /home/adity/Fast-dLLM-v2-extended/v2

# ==========================================
# CONFIGURATION - EDIT THESE VALUES
# ==========================================
LIMIT=100                    # Number of examples to evaluate (CHANGE THIS!)
MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_1.5B"
TASK="gsm8k"
BATCH_SIZE=1
MAX_NEW_TOKENS=512          # Increased from 128 to allow complete generation
BD_SIZE=8
SMALL_BLOCK_SIZE=4
THRESHOLD=0.9

# Thresholds to sweep
TOKEN_TAUS=(0.97 0.99)
LAYER_TAUS=(0.97 0.99)
# ==========================================

# Set environment variables
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create results directories
mkdir -p results/eval_outputs results/flops_stats

# Base model args
MODEL_ARGS_BASE="model_path=${MODEL_PATH},threshold=${THRESHOLD},max_new_tokens=${MAX_NEW_TOKENS},bd_size=${BD_SIZE},small_block_size=${SMALL_BLOCK_SIZE}"

echo "=========================================="
echo "Fast-dLLM v2 Compute-Skipping Evaluation"
echo "=========================================="
echo "Task: ${TASK}"
echo "Limit: ${LIMIT} examples"
echo "Model: ${MODEL_PATH}"
echo ""

# Function to extract accuracy from results.json
# Handles timestamped results files (results_*.json)
extract_accuracy() {
    local results_dir=$1
    local results_file=""
    
    # If it's a file, use it directly
    if [ -f "$results_dir" ]; then
        results_file="$results_dir"
    # If it's a directory, find the most recent results_*.json file
    elif [ -d "$results_dir" ]; then
        # Look for results.json first (if it exists)
        if [ -f "$results_dir/results.json" ]; then
            results_file="$results_dir/results.json"
        else
            # Find most recent results_*.json file
            results_file=$(find "$results_dir" -name "results_*.json" -type f 2>/dev/null | sort -r | head -1)
            # If still not found, check in subdirectories
            if [ -z "$results_file" ]; then
                results_file=$(find "$results_dir" -name "results_*.json" -type f 2>/dev/null | sort -r | head -1)
            fi
        fi
    fi
    
    if [ -n "$results_file" ] && [ -f "$results_file" ]; then
        python3 -c "
import json
import sys
try:
    with open('$results_file') as f:
        data = json.load(f)
    
    # Get task results
    task_results = data.get('results', {}).get('${TASK}', {})
    if not task_results:
        print('0.0')
        sys.exit(0)
    
    # Try multiple metric names (in order of preference for GSM8K)
    # The actual key is 'exact_match,flexible-extract' (with comma)
    metrics_to_try = [
        'exact_match,flexible-extract',  # Correct key format for GSM8K
        'exact_match',                    # Simple exact_match
        'acc',                            # Short form
        'acc_norm',                       # Normalized accuracy
    ]
    
    for metric in metrics_to_try:
        if metric in task_results:
            value = task_results[metric]
            if isinstance(value, (int, float)):
                print(value)
                sys.exit(0)
    
    # If no standard metric found, try to find any numeric value (excluding stderr)
    for key, value in task_results.items():
        if isinstance(value, (int, float)) and 'stderr' not in key.lower() and 'error' not in key.lower():
            print(value)
            sys.exit(0)
    
    print('0.0')
except Exception as e:
    print('0.0')
"
    else
        echo "0.0"
    fi
}

# Function to extract FLOPs reduction from stats.jsonl
# Reads the LAST line (most recent entry) since the file is appended to
extract_flops_reduction() {
    local stats_file=$1
    if [ -f "$stats_file" ]; then
        python3 -c "
import json
try:
    with open('$stats_file') as f:
        lines = [line.strip() for line in f if line.strip()]
        if lines:
            # Read the last line (most recent entry)
            data = json.loads(lines[-1])
            flops = data.get('flops_reduction', 0.0)
            print(flops)
        else:
            print(0.0)
except Exception as e:
    print(0.0)
"
    else
        echo "0.0"
    fi
}

# Function to clear GPU cache
clear_cache() {
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

# Results array (will be written to JSONL at the end)
RESULTS_FILE="results/gsm8k_sweep.jsonl"
> "$RESULTS_FILE"  # Clear results file

# ==========================================
# 1. BASELINE (No Skipping)
# ==========================================
echo "----------------------------------------"
echo "1. Running BASELINE (no skipping)..."
echo "----------------------------------------"

OUTPUT_DIR="results/baseline"
accelerate launch eval.py \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --model fast_dllm_v2 \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --model_args "${MODEL_ARGS_BASE}" \
    --output_path ${OUTPUT_DIR} \
    --limit ${LIMIT} > /tmp/baseline_output.log 2>&1

ACCURACY=$(extract_accuracy "${OUTPUT_DIR}")
echo "Baseline Accuracy: ${ACCURACY}"

# Save baseline result
echo "{\"task\": \"${TASK}\", \"token_tau\": 0.0, \"layer_tau\": 0.0, \"token_skip\": false, \"layer_skip\": false, \"accuracy\": ${ACCURACY}, \"flops_reduction\": 0.0}" >> "$RESULTS_FILE"

clear_cache
echo ""

# ==========================================
# 2. TOKEN SKIPPING SWEEP
# ==========================================
echo "----------------------------------------"
echo "2. Running TOKEN SKIPPING sweep..."
echo "----------------------------------------"

for tau in "${TOKEN_TAUS[@]}"; do
    echo "  Token tau=${tau}..."
    
    OUTPUT_DIR="results/token_${tau}"
    STATS_FILE="results/flops_stats/token_${tau}.jsonl"
    
    # Clear stats file before this run to ensure fresh results
    mkdir -p "$(dirname "$STATS_FILE")"
    > "$STATS_FILE"  # Clear the file
    
    accelerate launch eval.py \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot 0 \
        --confirm_run_unsafe_code \
        --model fast_dllm_v2 \
        --fewshot_as_multiturn \
        --apply_chat_template \
        --model_args "${MODEL_ARGS_BASE},token_skip=True,token_tau=${tau},layer_skip=False,skip_stats_path=${STATS_FILE}" \
        --output_path ${OUTPUT_DIR} \
        --limit ${LIMIT} > /tmp/token_${tau}_output.log 2>&1
    
    ACCURACY=$(extract_accuracy "${OUTPUT_DIR}")
    FLOPS=$(extract_flops_reduction "${STATS_FILE}")
    
    echo "    Accuracy: ${ACCURACY}, FLOPs Reduction: ${FLOPS}"
    
    # Save result
    echo "{\"task\": \"${TASK}\", \"token_tau\": ${tau}, \"layer_tau\": 0.0, \"token_skip\": true, \"layer_skip\": false, \"accuracy\": ${ACCURACY}, \"flops_reduction\": ${FLOPS}}" >> "$RESULTS_FILE"
    
    clear_cache
    sleep 2  # Brief pause between runs
done

echo ""

# ==========================================
# 3. LAYER SKIPPING SWEEP
# ==========================================
echo "----------------------------------------"
echo "3. Running LAYER SKIPPING sweep..."
echo "----------------------------------------"

for tau in "${LAYER_TAUS[@]}"; do
    echo "  Layer tau=${tau}..."
    
    OUTPUT_DIR="results/layer_${tau}"
    STATS_FILE="results/flops_stats/layer_${tau}.jsonl"
    
    # Clear stats file before this run to ensure fresh results
    mkdir -p "$(dirname "$STATS_FILE")"
    > "$STATS_FILE"  # Clear the file
    
    accelerate launch eval.py \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot 0 \
        --confirm_run_unsafe_code \
        --model fast_dllm_v2 \
        --fewshot_as_multiturn \
        --apply_chat_template \
        --model_args "${MODEL_ARGS_BASE},token_skip=False,layer_skip=True,layer_tau=${tau},skip_stats_path=${STATS_FILE}" \
        --output_path ${OUTPUT_DIR} \
        --limit ${LIMIT} > /tmp/layer_${tau}_output.log 2>&1
    
    ACCURACY=$(extract_accuracy "${OUTPUT_DIR}")
    FLOPS=$(extract_flops_reduction "${STATS_FILE}")
    
    echo "    Accuracy: ${ACCURACY}, FLOPs Reduction: ${FLOPS}"
    
    # Save result
    echo "{\"task\": \"${TASK}\", \"token_tau\": 0.0, \"layer_tau\": ${tau}, \"token_skip\": false, \"layer_skip\": true, \"accuracy\": ${ACCURACY}, \"flops_reduction\": ${FLOPS}}" >> "$RESULTS_FILE"
    
    clear_cache
    sleep 2
done

echo ""

# ==========================================
# 4. COMBINED SKIPPING (Optional)
# ==========================================
echo "----------------------------------------"
echo "4. Running COMBINED SKIPPING..."
echo "----------------------------------------"

COMBINED_TAU=0.97
echo "  Combined: token_tau=${COMBINED_TAU}, layer_tau=${COMBINED_TAU}..."

OUTPUT_DIR="results/combined_${COMBINED_TAU}_${COMBINED_TAU}"
STATS_FILE="results/flops_stats/combined_${COMBINED_TAU}_${COMBINED_TAU}.jsonl"

# Clear stats file before this run to ensure fresh results
mkdir -p "$(dirname "$STATS_FILE")"
> "$STATS_FILE"  # Clear the file

accelerate launch eval.py \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --model fast_dllm_v2 \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --model_args "${MODEL_ARGS_BASE},token_skip=True,token_tau=${COMBINED_TAU},layer_skip=True,layer_tau=${COMBINED_TAU},skip_stats_path=${STATS_FILE}" \
    --output_path ${OUTPUT_DIR} \
    --limit ${LIMIT} > /tmp/combined_output.log 2>&1

ACCURACY=$(extract_accuracy "${OUTPUT_DIR}")
FLOPS=$(extract_flops_reduction "${STATS_FILE}")

echo "    Accuracy: ${ACCURACY}, FLOPs Reduction: ${FLOPS}"

# Save result
echo "{\"task\": \"${TASK}\", \"token_tau\": ${COMBINED_TAU}, \"layer_tau\": ${COMBINED_TAU}, \"token_skip\": true, \"layer_skip\": true, \"accuracy\": ${ACCURACY}, \"flops_reduction\": ${FLOPS}}" >> "$RESULTS_FILE"

clear_cache
echo ""

# ==========================================
# 5. SUMMARY AND PLOT GENERATION
# ==========================================
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo ""

# Display summary table
echo "Summary of Results:"
echo "-------------------"
echo "Configuration                    | Accuracy | FLOPs Reduction"
echo "---------------------------------|----------|----------------"
echo "Baseline                         | $(printf "%8.3f" $(extract_accuracy "results/baseline")) | $(printf "%15.4f" 0.0)"

for tau in "${TOKEN_TAUS[@]}"; do
    acc=$(extract_accuracy "results/token_${tau}")
    flops=$(extract_flops_reduction "results/flops_stats/token_${tau}.jsonl")
    echo "Token skip (tau=${tau})           | $(printf "%8.3f" ${acc}) | $(printf "%15.4f" ${flops})"
done

for tau in "${LAYER_TAUS[@]}"; do
    acc=$(extract_accuracy "results/layer_${tau}")
    flops=$(extract_flops_reduction "results/flops_stats/layer_${tau}.jsonl")
    echo "Layer skip (tau=${tau})           | $(printf "%8.3f" ${acc}) | $(printf "%15.4f" ${flops})"
done

acc=$(extract_accuracy "results/combined_${COMBINED_TAU}_${COMBINED_TAU}")
flops=$(extract_flops_reduction "results/flops_stats/combined_${COMBINED_TAU}_${COMBINED_TAU}.jsonl")
echo "Combined (tau=${COMBINED_TAU})         | $(printf "%8.3f" ${acc}) | $(printf "%15.4f" ${flops})"

echo ""
echo "Generating plot..."

# Generate plot
python3 scripts/plot_tradeoff.py \
    --input "${RESULTS_FILE}" \
    --output "results/gsm8k_tradeoff.png" \
    --task "GSM8K"

if [ -f "results/gsm8k_tradeoff.png" ]; then
    echo "✓ Plot saved to: results/gsm8k_tradeoff.png"
else
    echo "⚠ Warning: Plot generation may have failed"
fi

echo ""
echo "=========================================="
echo "All done! Check:"
echo "  - Results: ${RESULTS_FILE}"
echo "  - Plot: results/gsm8k_tradeoff.png"
echo "=========================================="