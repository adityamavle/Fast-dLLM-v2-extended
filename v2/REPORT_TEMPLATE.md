# Compute-Skipping Policies for Fast-dLLM v2: Implementation Report

**Author:** [Your Name]  
**Date:** [Date]  
**Task:** Implement token-level and layer-level compute-skipping policies with accuracy vs FLOPs-reduction curves

---

## 1. Executive Summary

This report presents the implementation of two compute-skipping policies for Fast-dLLM v2:
- **Token-level skipping**: Skips computing tokens across denoising steps when hidden states are stable
- **Layer-level skipping**: Skips transformer layers within a denoising step when inputs are similar

Both policies use cosine similarity thresholds to determine when computation can be safely skipped, enabling FLOPs reduction while maintaining model accuracy.

---

## 2. Implementation Overview

### 2.1 Architecture

The implementation consists of:

1. **Skip Policy Modules** (`v2/skip/`):
   - `token_policy.py`: Token-level skipping logic with probe computation
   - `layer_policy.py`: Layer-level skipping with similarity comparison
   - `stats.py`: FLOPs reduction statistics tracking
   - `model_hooks.py`: Transformer layer wrapping for skipping

2. **Integration Points**:
   - `generation_functions.py`: Modified `batch_sample()` and `mdm_sample_with_visualization()` to support skipping
   - `eval.py`: Added skip parameters to evaluation harness

3. **Evaluation Tools**:
   - `scripts/sweep_skip_eval.py`: Automated threshold sweeping
   - `scripts/plot_tradeoff.py`: Accuracy vs FLOPs reduction visualization

### 2.2 Key Design Decisions

- **Probe-based token skipping**: Uses first layer output as a cheap probe to avoid full forward pass for similarity comparison
- **Cosine similarity**: Standard metric for measuring hidden state stability
- **FLOPs proxy**: Uses `active_tokens × executed_layers` as a proxy for actual FLOPs reduction
- **Threshold-based control**: Allows fine-tuning tradeoff between accuracy and FLOPs reduction

---

## 3. Methodology

### 3.1 Token-Level Skipping

**Algorithm:**
1. At each denoising step `t`, compute probe hidden state `h_probe` (first layer output)
2. Compare `h_probe` with previous step's hidden state `h_prev` using cosine similarity
3. If `cos_sim(h_probe, h_prev) >= tau_token`, skip computing that token (reuse `h_prev`)
4. Track number of active (non-skipped) tokens per step

**Implementation Location:** `v2/skip/token_policy.py`, integrated in `generation_functions.py` lines 130-165

### 3.2 Layer-Level Skipping

**Algorithm:**
1. Before each transformer layer `l`, compare input `x_in` with previous layer's input `x_prev_in`
2. Compute mean cosine similarity: `mean_token_cos(x_in, x_prev_in)`
3. If `mean_sim >= tau_layer`, skip the layer (identity pass: `x_out = x_in`)
4. Track number of executed layers per step

**Implementation Location:** `v2/skip/layer_policy.py`, integrated via `model_hooks.py`

### 3.3 FLOPs Reduction Calculation

```
FLOPs_reduction = 1 - (sum(active_tokens × executed_layers) / (total_steps × seq_len × total_layers))
```

Where:
- `active_tokens`: Tokens computed after token skipping
- `executed_layers`: Layers executed after layer skipping
- `total_steps`: Total denoising steps
- `seq_len`: Sequence length
- `total_layers`: Total transformer layers

---

## 4. Experimental Setup

### 4.1 Evaluation Tasks

- **Primary Task**: GSM8K (Grade School Math 8K)
- **Additional Tasks**: [List any other tasks tested]

### 4.2 Threshold Sweep

**Token skipping thresholds:** `[0.95, 0.97, 0.99, 0.995, 0.999]`  
**Layer skipping thresholds:** `[0.95, 0.97, 0.99, 0.995, 0.999]`

**Baseline:** No skipping (all tokens and layers computed)

### 4.3 Hardware/Software

- **Model**: Fast-dLLM v2 7B
- **Hardware**: [Your GPU/CPU specs]
- **Software**: Python 3.x, PyTorch, Accelerate, lm_eval

---

## 5. Results

### 5.1 Baseline Performance

| Task | Accuracy | FLOPs Reduction |
|------|----------|-----------------|
| GSM8K | [X.XXX] | 0.0% |

### 5.2 Token-Level Skipping Results

| Threshold (τ) | Accuracy | FLOPs Reduction |
|---------------|----------|-----------------|
| 0.95 | [X.XXX] | [X.XX]% |
| 0.97 | [X.XXX] | [X.XX]% |
| 0.99 | [X.XXX] | [X.XX]% |
| 0.995 | [X.XXX] | [X.XX]% |
| 0.999 | [X.XXX] | [X.XX]% |

### 5.3 Layer-Level Skipping Results

| Threshold (τ) | Accuracy | FLOPs Reduction |
|---------------|----------|-----------------|
| 0.95 | [X.XXX] | [X.XX]% |
| 0.97 | [X.XXX] | [X.XX]% |
| 0.99 | [X.XXX] | [X.XX]% |
| 0.995 | [X.XXX] | [X.XX]% |
| 0.999 | [X.XXX] | [X.XX]% |

### 5.4 Accuracy vs FLOPs Reduction Curves

**Include your generated plot here:**
- Figure 1: Accuracy vs FLOPs Reduction for GSM8K
  - Baseline (horizontal line)
  - Token-level skipping (blue curve)
  - Layer-level skipping (red curve)

**Analysis:**
- [Describe the tradeoff curve]
- [Identify optimal threshold points]
- [Compare token vs layer skipping effectiveness]

---

## 6. Analysis and Discussion

### 6.1 Key Findings

1. **Token-Level Skipping:**
   - [Your observations]
   - Example: "At τ=0.99, achieves X% FLOPs reduction with only Y% accuracy drop"

2. **Layer-Level Skipping:**
   - [Your observations]
   - Example: "More aggressive than token skipping, achieving higher FLOPs reduction"

3. **Tradeoff Analysis:**
   - [Discuss the accuracy-FLOPs tradeoff]
   - [Identify sweet spots]

### 6.2 Comparison

| Policy | Max FLOPs Reduction | Accuracy Drop at Max Reduction |
|--------|---------------------|--------------------------------|
| Token-level | [X]% | [Y]% |
| Layer-level | [X]% | [Y]% |

### 6.3 Limitations and Future Work

- **Current limitations:**
  - Probe computation adds overhead
  - FLOPs reduction is a proxy metric
  - Threshold tuning required per task

- **Future improvements:**
  - Adaptive threshold selection
  - More efficient probe computation
  - Combined policy optimization

---

## 7. Code Structure

### 7.1 File Organization

```
v2/
├── skip/                    # Skip policy modules
│   ├── token_policy.py     # Token-level skipping
│   ├── layer_policy.py     # Layer-level skipping
│   ├── stats.py            # FLOPs statistics
│   └── model_hooks.py      # Layer wrapping
├── scripts/
│   ├── sweep_skip_eval.py  # Evaluation sweeps
│   └── plot_tradeoff.py   # Plotting script
├── generation_functions.py # Modified generation
└── eval.py                 # Modified evaluation harness
```

### 7.2 Key Functions

**Token Skipping:**
- `TokenSkipPolicy.compute_probe_hidden()`: Computes cheap probe
- `TokenSkipPolicy.compute_skip_mask()`: Determines which tokens to skip

**Layer Skipping:**
- `LayerSkipPolicy.should_skip_layer()`: Decides if layer should be skipped
- `apply_layer_skipping()`: Wraps transformer layers

**Statistics:**
- `SkipStats.record_step()`: Records per-step stats
- `SkipStats.compute_flops_reduction()`: Calculates FLOPs reduction

---

## 8. Usage Instructions

### 8.1 Running Evaluation

```bash
# Baseline
accelerate launch eval.py \
    --tasks gsm8k \
    --batch_size 32 \
    --model fast_dllm_v2 \
    --model_args "model_path=Efficient-Large-Model/Fast_dLLM_v2_7B,threshold=0.9"

# With token skipping
accelerate launch eval.py \
    --tasks gsm8k \
    --model fast_dllm_v2 \
    --model_args "model_path=Efficient-Large-Model/Fast_dLLM_v2_7B,token_skip=True,token_tau=0.99,skip_stats_path=./stats.jsonl"
```

### 8.2 Running Sweep

```bash
python scripts/sweep_skip_eval.py \
    --task gsm8k \
    --token_taus 0.95 0.97 0.99 0.995 0.999 \
    --layer_taus 0.95 0.97 0.99 0.995 0.999 \
    --results_file ./results/gsm8k_sweep.jsonl
```

### 8.3 Generating Plots

```bash
python scripts/plot_tradeoff.py \
    --input ./results/gsm8k_sweep.jsonl \
    --output ./results/gsm8k_tradeoff.png \
    --task "GSM8K"
```

---

## 9. Conclusion

[Summarize key achievements]
- Successfully implemented both skipping policies
- Achieved X% FLOPs reduction with Y% accuracy preservation
- Generated comprehensive accuracy vs FLOPs reduction curves

[Final thoughts on the implementation and results]

---

## 10. Appendix

### A. Complete Results Data

[Include full results JSONL or link to results file]

### B. Additional Figures

[Any other relevant plots or visualizations]

### C. Code Repository

[Link to your implementation if hosted]

---

## References

- Fast-dLLM v2: [Paper/Repo link]
- Original task requirements: [Reference]
