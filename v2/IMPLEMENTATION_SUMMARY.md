# Compute-Skipping Implementation Summary

## Overview

This implementation adds two compute-skipping policies to Fast-dLLM v2:

1. **Token-level skipping**: Skips computing tokens across denoising steps when hidden states are stable
2. **Layer-level skipping**: Skips transformer layers within a denoising step when inputs are similar

## Files Created/Modified

### New Files

1. **`v2/skip/`** - Skip policy module
   - `__init__.py` - Module exports
   - `token_policy.py` - Token-level skipping logic
   - `layer_policy.py` - Layer-level skipping logic
   - `stats.py` - FLOPs reduction statistics
   - `model_hooks.py` - Model layer wrapping for skipping
   - `README.md` - Module documentation

2. **`v2/scripts/`** - Evaluation and plotting scripts
   - `sweep_skip_eval.py` - Sweep different thresholds and collect results
   - `plot_tradeoff.py` - Plot accuracy vs FLOPs reduction curves

### Modified Files

1. **`v2/generation_functions.py`**
   - Added skip parameters to `batch_sample()` and `mdm_sample_with_visualization()`
   - Integrated token and layer skipping logic into denoising loops
   - Added FLOPs stats collection

2. **`v2/eval.py`**
   - Added skip parameters to `Fast_dLLM_v2EvalHarness.__init__()`
   - Modified `generate_until()` to pass skip parameters and collect stats
   - Added stats saving functionality

## How It Works

### Token-Level Skipping

1. At each denoising step, compute a "probe" hidden state (first layer output)
2. Compare with previous step's hidden state using cosine similarity
3. If similarity >= `token_tau`, skip computing that token (reuse previous output)
4. Track active tokens for FLOPs calculation

### Layer-Level Skipping

1. Before each transformer layer, compare input with previous layer's input
2. If mean cosine similarity >= `layer_tau`, skip the layer (identity pass)
3. Track executed layers for FLOPs calculation

### FLOPs Reduction

Calculated as:
```
FLOPs_reduction = 1 - (sum(active_tokens × executed_layers) / (total_steps × seq_len × total_layers))
```

## Usage

### Basic Evaluation with Skipping

```bash
cd v2
accelerate launch eval.py \
    --tasks gsm8k \
    --batch_size 32 \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --model fast_dllm_v2 \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --model_args "model_path=Efficient-Large-Model/Fast_dLLM_v2_7B,token_skip=True,token_tau=0.99,layer_skip=False,skip_stats_path=./stats.jsonl,threshold=0.9"
```

### Running a Sweep

```bash
cd v2
python scripts/sweep_skip_eval.py \
    --task gsm8k \
    --output_dir ./results/eval_outputs \
    --stats_dir ./results/flops_stats \
    --results_file ./results/gsm8k_sweep.jsonl \
    --token_taus 0.95 0.97 0.99 0.995 0.999 \
    --layer_taus 0.95 0.97 0.99 0.995 0.999
```

### Plotting Results

```bash
cd v2
python scripts/plot_tradeoff.py \
    --input ./results/gsm8k_sweep.jsonl \
    --output ./results/gsm8k_tradeoff.png \
    --task "GSM8K"
```

## Parameters

- `token_skip` (bool): Enable token-level skipping
- `token_tau` (float): Token skipping threshold (0.95-0.999, higher = more conservative)
- `layer_skip` (bool): Enable layer-level skipping  
- `layer_tau` (float): Layer skipping threshold (0.95-0.999, higher = more conservative)
- `skip_stats_path` (str): Path to save FLOPs statistics JSONL file

## Expected Outputs

1. **Evaluation results**: JSON files with accuracy metrics in `--output_dir`
2. **FLOPs stats**: JSONL files with per-run statistics in `--stats_dir`
3. **Combined results**: JSONL file with accuracy + FLOPs reduction
4. **Plot**: PNG file showing accuracy vs FLOPs reduction tradeoff curve

## Testing Checklist

- [ ] Baseline runs without skipping (verify accuracy matches expected)
- [ ] Token skipping reduces FLOPs as threshold increases
- [ ] Layer skipping reduces FLOPs as threshold increases
- [ ] Accuracy decreases as FLOPs reduction increases (tradeoff curve)
- [ ] Combined skipping shows better FLOPs reduction than individual policies
- [ ] Stats are saved correctly to JSONL files
- [ ] Plot shows clear tradeoff curves

## Notes

- The implementation uses cosine similarity for stability detection
- Token skipping uses a "probe" (first layer) to avoid full forward pass
- Layer skipping wraps transformer layers to intercept forward calls
- FLOPs reduction is a proxy metric based on active tokens × executed layers
- Higher thresholds (0.99+) are more conservative and preserve accuracy better

