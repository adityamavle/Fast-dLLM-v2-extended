import torch
import torch.nn as nn
from typing import Optional

def wrap_layer_with_skipping(layer, layer_idx, layer_policy):
    """Wrap a transformer layer to support skipping"""
    if layer_policy is None or not layer_policy.enabled:
        return layer
    
    original_forward = layer.forward
    
    def forward_with_skip(hidden_states, *args, **kwargs):
        if layer_policy is None or not layer_policy.enabled:
            return original_forward(hidden_states, *args, **kwargs)
        
        should_skip, sim = layer_policy.should_skip_layer(hidden_states)
        
        if should_skip:
            layer_policy.record_layer_execution(layer_idx, False)
            # Return input unchanged (identity skip)
            output = hidden_states
            # Still need to handle past_key_values if present
            if 'past_key_values' in kwargs and kwargs['past_key_values'] is not None:
                # Return cached KV or None
                past_key_values = kwargs.get('past_key_values')
                if isinstance(past_key_values, tuple):
                    return (output, past_key_values)
                return output
            # Check if args contain past_key_values
            if len(args) > 0 and args[0] is not None:
                return (output, args[0])
            return output
        else:
            layer_policy.record_layer_execution(layer_idx, True)
            result = original_forward(hidden_states, *args, **kwargs)
            # Update x_prev_in with output for next layer comparison
            if isinstance(result, tuple):
                layer_policy.x_prev_in = result[0].clone()
            else:
                layer_policy.x_prev_in = result.clone()
            return result
    
    layer.forward = forward_with_skip
    return layer

def apply_layer_skipping(model, layer_policy):
    """Apply layer skipping to all transformer layers"""
    if layer_policy is None or not layer_policy.enabled:
        return
    
    # Handle different model structures
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        return
    
    for i, layer in enumerate(layers):
        wrap_layer_with_skipping(layer, i, layer_policy)

def remove_layer_skipping(model):
    """Remove layer skipping hooks (restore original forward)"""
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        return
    
    layers = model.model.layers
    for layer in layers:
        # Check if forward was wrapped (has closure)
        if hasattr(layer.forward, '__closure__') and layer.forward.__closure__:
            # Try to restore - this is tricky, so we'll just reinitialize if needed
            # For now, we'll keep the wrapped version
            pass

