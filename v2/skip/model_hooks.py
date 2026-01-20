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
        
        # CRITICAL FIX: Store current INPUT before comparison
        # We compare Layer L's input to Layer L-1's input (not output)
        # This prevents cascade skipping when previous layer was skipped
        current_input = hidden_states.clone()
        
        # Compare current input to previous layer's input
        should_skip, sim = layer_policy.should_skip_layer(hidden_states)
        
        # SAFEGUARD 1: Never skip the first 2 layers to ensure basic processing
        # This prevents over-aggressive skipping that breaks generation
        if layer_idx < 2:
            should_skip = False
        
        # SAFEGUARD 2: Prevent too many consecutive skips (max 3-4 consecutive)
        # This prevents cascade skipping that breaks generation
        MAX_CONSECUTIVE_SKIPS = 3
        if should_skip and layer_policy.consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
            should_skip = False
        
        if should_skip:
            layer_policy.record_layer_execution(layer_idx, False)
            layer_policy.consecutive_skips += 1
            
            # Check if cache needs updating
            use_cache = kwargs.get('use_cache', False)
            update_past_key_values = kwargs.get('update_past_key_values', False)
            
            if use_cache and update_past_key_values:
                # PARTIAL EXECUTION: Run attention to update cache, skip MLP
                # This ensures cache consistency while still saving MLP computation (~60-70% of layer FLOPs)
                
                # Store original input for identity skip
                original_input = hidden_states.clone()
                
                # Run attention path to update cache
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
                
                # Run attention (this updates the KV cache)
                hidden_states = layer.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=kwargs.get('attention_mask'),
                    position_ids=kwargs.get('position_ids'),
                    past_key_value=kwargs.get('past_key_value'),
                    use_cache=use_cache,
                    cache_position=kwargs.get('cache_position'),
                    position_embeddings=kwargs.get('position_embeddings'),
                    update_past_key_values=update_past_key_values,
                    use_block_cache=kwargs.get('use_block_cache', False),
                    block_past_key_values=kwargs.get('block_past_key_values'),
                    replace_position=kwargs.get('replace_position'),
                )
                
                # Skip MLP and return original input (identity skip)
                # This maintains cache consistency while saving computation
                output = original_input
            else:
                # No cache updating needed: full skip (return input unchanged)
                if isinstance(hidden_states, tuple):
                    output = hidden_states[0]
                else:
                    output = hidden_states
                
                # Ensure output is a tensor
                if not isinstance(output, torch.Tensor):
                    if isinstance(output, tuple):
                        output = output[0]
                    else:
                        raise ValueError(f"Unexpected output type when skipping layer: {type(output)}")
            
            # Update x_prev_in to current INPUT (not output) for next layer's comparison
            # This ensures we always compare input-to-input between adjacent layers
            layer_policy.x_prev_in = current_input
            
            return output
        else:
            layer_policy.record_layer_execution(layer_idx, True)
            layer_policy.consecutive_skips = 0  # Reset consecutive skips counter
            result = original_forward(hidden_states, *args, **kwargs)
            # Update x_prev_in to current INPUT (not output) for next layer's comparison
            # This ensures we always compare input-to-input between adjacent layers
            layer_policy.x_prev_in = current_input
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

def register_hidden_state_hook(model, token_policy):
    """
    Register a forward hook to capture final hidden states for token skipping.
    This allows us to store the final hidden states from each forward pass.
    """
    if token_policy is None or not token_policy.enabled:
        return None
    
    def hook_fn(module, input, output):
        """Hook to capture final hidden states before logits"""
        # The output is typically a tuple or BaseModelOutput
        # We want the hidden_states (first element)
        if isinstance(output, tuple):
            hidden_states = output[0]
        elif hasattr(output, 'last_hidden_state'):
            hidden_states = output.last_hidden_state
        elif hasattr(output, 'hidden_states') and output.hidden_states is not None:
            # If hidden_states is a tuple, get the last one
            if isinstance(output.hidden_states, tuple) and len(output.hidden_states) > 0:
                hidden_states = output.hidden_states[-1]
            else:
                hidden_states = output.hidden_states
        else:
            return
        
        # Store in token policy for next step
        if hidden_states is not None:
            token_policy.h_prev_final = hidden_states.detach().clone()
    
    # Register hook on the model's base model (before lm_head)
    if hasattr(model, 'model'):
        hook_handle = model.model.register_forward_hook(hook_fn)
        return hook_handle
    elif hasattr(model, 'transformer'):
        hook_handle = model.transformer.register_forward_hook(hook_fn)
        return hook_handle
    
    return None

def remove_hidden_state_hook(hook_handle):
    """Remove the hidden state hook"""
    if hook_handle is not None:
        hook_handle.remove()
