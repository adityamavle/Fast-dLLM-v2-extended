#!/bin/bash
# run_comprehensive_eval.sh
# Comprehensive evaluation script for token and layer skipping
# Generates results table and accuracy vs FLOPs plot

set -e

cd /home/hice1/amavle3/Fast-dLLM-v2-extended/v2

# ==========================================
# CONFIGURATION - User Specified Parameters
# ==========================================
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Baseline parameters (as specified by user)
TASK="gsm8k"
BATCH_SIZE="32"
NUM_FEWSHOT="5"
MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_7B"
THRESHOLD="1"
LIMIT="100"

# Tau values for sweep
TOKEN_TAUS=(0.90 0.97 0.99)
LAYER_TAUS=(0.90 0.97 0.99)

# Create results directories
mkdir -p results/eval_outputs results/flops_stats

RESULTS_FILE="results/gsm8k_comprehensive.jsonl"
> "$RESULTS_FILE"  # Clear results file

echo "=========================================="
echo "Fast-dLLM v2 Comprehensive Evaluation"
echo "=========================================="
echo "Task: ${TASK}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Num Fewshot: ${NUM_FEWSHOT}"
echo "Limit: ${LIMIT} examples"
echo "Model: ${MODEL_PATH}"
echo ""

# Function to extract accuracy from results
extract_accuracy() {
    local results_dir=$1
    if [ -f "$results_dir" ]; then
        results_file="$results_dir"
    elif [ -d "$results_dir" ]; then
        results_file=$(find "$results_dir" -name "results_*.json" -type f 2>/dev/null | sort -r | head -1)
        [ -z "$results_file" ] && results_file=$(find "$results_dir" -name "results.json" -type f 2>/dev/null | head -1)
    fi
    
    if [ -n "$results_file" ] && [ -f "$results_file" ]; then
        python3 -c "
import json, sys
try:
    with open('$results_file') as f:
        data = json.load(f)
    task_results = data.get('results', {}).get('${TASK}', {})
    metrics = ['exact_match,flexible-extract', 'exact_match', 'acc', 'acc_norm']
    for metric in metrics:
        if metric in task_results and isinstance(task_results[metric], (int, float)):
            print(task_results[metric])
            sys.exit(0)
    for key, value in task_results.items():
        if isinstance(value, (int, float)) and 'stderr' not in key.lower():
            print(value)
            sys.exit(0)
    print('0.0')
except: print('0.0')
"
    else
        echo "0.0"
    fi
}

# Function to extract FLOPs reduction from stats.jsonl
extract_flops_reduction() {
    local stats_file=$1
    if [ -f "$stats_file" ]; then
        python3 -c "
import json
try:
    with open('$stats_file') as f:
        lines = [line.strip() for line in f if line.strip()]
        if lines:
            data = json.loads(lines[-1])
            print(data.get('flops_reduction', 0.0))
        else:
            print(0.0)
except: print(0.0)
"
    else
        echo "0.0"
    fi
}

clear_cache() {
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

# ==========================================
# 1. BASELINE (No Skipping)
# ==========================================
echo "----------------------------------------"
echo "1. BASELINE (No Skipping)"
echo "----------------------------------------"

OUTPUT_DIR="results/eval_outputs/baseline"
mkdir -p "$OUTPUT_DIR"

accelerate launch eval.py \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot ${NUM_FEWSHOT} \
    --confirm_run_unsafe_code \
    --model fast_dllm_v2 \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --model_args "model_path=${MODEL_PATH},threshold=${THRESHOLD},show_speed=True" \
    --output_path ${OUTPUT_DIR} \
    --limit ${LIMIT} > /tmp/baseline.log 2>&1

ACCURACY=$(extract_accuracy "${OUTPUT_DIR}")
echo "Baseline Accuracy: ${ACCURACY}"
echo "{\"task\": \"${TASK}\", \"method\": \"baseline\", \"token_tau\": 0.0, \"layer_tau\": 0.0, \"accuracy\": ${ACCURACY}, \"flops_reduction\": 0.0}" >> "$RESULTS_FILE"

clear_cache
echo ""

# ==========================================
# 2. TOKEN-LEVEL SKIPPING
# ==========================================
echo "----------------------------------------"
echo "2. TOKEN-LEVEL SKIPPING"
echo "----------------------------------------"

for tau in "${TOKEN_TAUS[@]}"; do
    echo "  tau=${tau}..."
    
    OUTPUT_DIR="results/eval_outputs/token_${tau}"
    STATS_FILE="results/flops_stats/token_${tau}.jsonl"
    mkdir -p "$OUTPUT_DIR" "$(dirname "$STATS_FILE")"
    > "$STATS_FILE"  # Clear
    
    accelerate launch eval.py \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot ${NUM_FEWSHOT} \
        --confirm_run_unsafe_code \
        --model fast_dllm_v2 \
        --fewshot_as_multiturn \
        --apply_chat_template \
        --model_args "model_path=${MODEL_PATH},threshold=${THRESHOLD},show_speed=True,token_skip=True,token_tau=${tau},layer_skip=False,skip_stats_path=${STATS_FILE}" \
        --output_path ${OUTPUT_DIR} \
        --limit ${LIMIT} > /tmp/token_${tau}.log 2>&1
    
    ACCURACY=$(extract_accuracy "${OUTPUT_DIR}")
    FLOPS=$(extract_flops_reduction "${STATS_FILE}")
    echo "    Accuracy: ${ACCURACY}, FLOPs Reduction: ${FLOPS}"
    echo "{\"task\": \"${TASK}\", \"method\": \"token_skip\", \"token_tau\": ${tau}, \"layer_tau\": 0.0, \"accuracy\": ${ACCURACY}, \"flops_reduction\": ${FLOPS}}" >> "$RESULTS_FILE"
    
    clear_cache
    sleep 2
done

echo ""

# ==========================================
# 3. LAYER-LEVEL SKIPPING
# ==========================================
echo "----------------------------------------"
echo "3. LAYER-LEVEL SKIPPING"
echo "----------------------------------------"

for tau in "${LAYER_TAUS[@]}"; do
    echo "  tau=${tau}..."
    
    OUTPUT_DIR="results/eval_outputs/layer_${tau}"
    STATS_FILE="results/flops_stats/layer_${tau}.jsonl"
    mkdir -p "$OUTPUT_DIR" "$(dirname "$STATS_FILE")"
    > "$STATS_FILE"  # Clear
    
    accelerate launch eval.py \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot ${NUM_FEWSHOT} \
        --confirm_run_unsafe_code \
        --model fast_dllm_v2 \
        --fewshot_as_multiturn \
        --apply_chat_template \
        --model_args "model_path=${MODEL_PATH},threshold=${THRESHOLD},show_speed=True,token_skip=False,layer_skip=True,layer_tau=${tau},skip_stats_path=${STATS_FILE}" \
        --output_path ${OUTPUT_DIR} \
        --limit ${LIMIT} > /tmp/layer_${tau}.log 2>&1
    
    ACCURACY=$(extract_accuracy "${OUTPUT_DIR}")
    FLOPS=$(extract_flops_reduction "${STATS_FILE}")
    echo "    Accuracy: ${ACCURACY}, FLOPs Reduction: ${FLOPS}"
    echo "{\"task\": \"${TASK}\", \"method\": \"layer_skip\", \"token_tau\": 0.0, \"layer_tau\": ${tau}, \"accuracy\": ${ACCURACY}, \"flops_reduction\": ${FLOPS}}" >> "$RESULTS_FILE"
    
    clear_cache
    sleep 2
done

echo ""

# ==========================================
# 4. SUMMARY AND VISUALIZATION
# ==========================================
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results file: ${RESULTS_FILE}"
echo ""

echo "Summary Table:"
echo "---------------------------------------------------"
printf "%-25s | %-10s | %-15s\n" "Method" "Accuracy" "FLOPs Reduction"
echo "---------------------------------------------------"

# Baseline
acc=$(extract_accuracy "results/eval_outputs/baseline")
printf "%-25s | %-10.4f | %-15.4f\n" "Baseline" "$acc" "0.0000"

# Token skipping
for tau in "${TOKEN_TAUS[@]}"; do
    acc=$(extract_accuracy "results/eval_outputs/token_${tau}")
    flops=$(extract_flops_reduction "results/flops_stats/token_${tau}.jsonl")
    printf "%-25s | %-10.4f | %-15.4f\n" "Token Skip (τ=$tau)" "$acc" "$flops"
done

# Layer skipping
for tau in "${LAYER_TAUS[@]}"; do
    acc=$(extract_accuracy "results/eval_outputs/layer_${tau}")
    flops=$(extract_flops_reduction "results/flops_stats/layer_${tau}.jsonl")
    printf "%-25s | %-10.4f | %-15.4f\n" "Layer Skip (τ=$tau)" "$acc" "$flops"
done

echo ""
echo "Generating plot..."

# Generate plot
python3 scripts/plot_tradeoff.py \
    --input "${RESULTS_FILE}" \
    --output "results/accuracy_vs_flops.png" \
    --task "GSM8K"

if [ -f "results/accuracy_vs_flops.png" ]; then
    echo "✓ Plot saved to: results/accuracy_vs_flops.png"
else
    echo "⚠ Warning: Plot generation may have failed"
fi

echo ""
echo "=========================================="
echo "✓ Evaluation pipeline complete!"
echo "=========================================="