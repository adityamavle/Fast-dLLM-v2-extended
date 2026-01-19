# Compute-Skipping Policies for Fast-dLLM v2

This module implements two compute-skipping policies to reduce FLOPs during inference:

1. **Token-level skipping**: Skips computing tokens whose hidden states are stable across denoising steps
2. **Layer-level skipping**: Skips transformer layers when input hidden states are similar to previous layer inputs

## Files

- `token_policy.py`: Token-level skipping implementation
- `layer_policy.py`: Layer-level skipping implementation  
- `stats.py`: FLOPs reduction statistics tracking
- `model_hooks.py`: Model layer wrapping for skipping
- `__init__.py`: Module exports

## Usage

The skipping policies are integrated into the generation functions and can be enabled via parameters:

```python
from skip import TokenSkipPolicy, LayerSkipPolicy, SkipStats

# In generation call:
generated_ids = model.mdm_sample(
    input_ids,
    tokenizer=tokenizer,
    # ... other params ...
    token_skip_enabled=True,
    token_tau=0.99,  # Threshold for token skipping
    layer_skip_enabled=True,
    layer_tau=0.99,  # Threshold for layer skipping
    skip_stats=skip_stats,  # Optional: track FLOPs reduction
)
```

## Parameters

- `token_tau`: Cosine similarity threshold for token skipping (0.95-0.999)
- `layer_tau`: Cosine similarity threshold for layer skipping (0.95-0.999)
- Higher thresholds = more conservative skipping = less FLOPs reduction but better accuracy
- Lower thresholds = more aggressive skipping = more FLOPs reduction but potential accuracy loss

## FLOPs Reduction Calculation

FLOPs reduction is computed as:
```
FLOPs_reduction = 1 - (sum(active_tokens * executed_layers) / (total_steps * seq_len * total_layers))
```

Where:
- `active_tokens`: Number of tokens computed (after token skipping)
- `executed_layers`: Number of layers executed (after layer skipping)
- `total_steps`: Total denoising steps
- `seq_len`: Sequence length
- `total_layers`: Total transformer layers

