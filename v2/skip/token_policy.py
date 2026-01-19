import torch
import torch.nn.functional as F

class TokenSkipPolicy:
    """Token-level skipping across denoising steps"""
    
    def __init__(self, tau_token=0.99, enabled=True, probe_layer_idx=0):
        self.tau_token = tau_token
        self.enabled = enabled
        self.probe_layer_idx = probe_layer_idx
        self.h_prev = None  # Previous step hidden states [B, S, H]
        
    def reset(self):
        """Reset state between sequences"""
        self.h_prev = None
    
    def compute_probe_hidden(self, model, input_ids, past_key_values=None):
        """
        Compute cheap probe hidden state (first layer output)
        Returns: [B, S, H] hidden states
        """
        # Get embeddings - handle different model structures
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            inputs_embeds = model.model.embed_tokens(input_ids)
            layers = model.model.layers
        elif hasattr(model, 'embed_tokens'):
            inputs_embeds = model.embed_tokens(input_ids)
            layers = model.layers if hasattr(model, 'layers') else []
        else:
            # Fallback: return embeddings as probe
            return model.model.embed_tokens(input_ids) if hasattr(model, 'model') else inputs_embeds
        
        # Run through first layer only (cheap probe)
        hidden_states = inputs_embeds
        if len(layers) > self.probe_layer_idx:
            layer = layers[self.probe_layer_idx]
            # Get layer output (simplified - just run first layer)
            try:
                layer_outputs = layer(
                    hidden_states,
                    past_key_values=past_key_values[self.probe_layer_idx] if past_key_values and len(past_key_values) > self.probe_layer_idx and past_key_values[self.probe_layer_idx] is not None else None,
                    use_cache=False
                )
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs
            except Exception:
                # If layer forward fails, return embeddings
                pass
        
        return hidden_states
    
    def compute_skip_mask(self, h_current, h_prev):
        """
        Compute boolean mask for tokens to skip
        Returns: [B, S] boolean mask (True = skip)
        """
        if h_prev is None or not self.enabled:
            return torch.zeros(h_current.shape[:2], dtype=torch.bool, device=h_current.device)
        
        # Ensure same sequence length
        min_seq_len = min(h_current.shape[1], h_prev.shape[1])
        h_curr = h_current[:, :min_seq_len, :]
        h_prev_trunc = h_prev[:, :min_seq_len, :]
        
        # Compute cosine similarity per token
        # h_current: [B, S, H], h_prev: [B, S, H]
        h_curr_norm = F.normalize(h_curr, p=2, dim=-1)  # [B, S, H]
        h_prev_norm = F.normalize(h_prev_trunc, p=2, dim=-1)     # [B, S, H]
        
        cos_sim = (h_curr_norm * h_prev_norm).sum(dim=-1)  # [B, S]
        
        # Skip if similarity >= threshold
        skip_mask = cos_sim >= self.tau_token
        
        # Pad mask if needed
        if h_current.shape[1] > min_seq_len:
            padding = torch.zeros(h_current.shape[0], h_current.shape[1] - min_seq_len, 
                                dtype=torch.bool, device=h_current.device)
            skip_mask = torch.cat([skip_mask, padding], dim=1)
        
        return skip_mask
    
    def pack_active_tokens(self, hidden_states, skip_mask):
        """
        Pack only active (non-skipped) tokens for efficient computation
        Returns: packed_hidden, active_indices, active_mask
        """
        B, S, H = hidden_states.shape
        active_mask = ~skip_mask  # [B, S]
        
        # Flatten and pack
        active_indices = active_mask.nonzero(as_tuple=False)  # [N_active, 2] (batch_idx, seq_idx)
        packed_hidden = hidden_states[active_mask]  # [N_active, H]
        
        return packed_hidden, active_indices, active_mask
    
    def scatter_back(self, packed_hidden, skip_mask, original_shape, active_indices):
        """
        Scatter packed hidden states back to original positions
        """
        B, S, H = original_shape
        device = packed_hidden.device
        
        # Initialize output with previous hidden states for skipped tokens
        output = torch.zeros(B, S, H, device=device, dtype=packed_hidden.dtype)
        
        # Scatter active tokens
        if len(active_indices) > 0:
            batch_idx = active_indices[:, 0]
            seq_idx = active_indices[:, 1]
            output[batch_idx, seq_idx] = packed_hidden
        
        # For skipped tokens, use previous hidden states
        if self.h_prev is not None:
            # Ensure same shape
            min_seq = min(S, self.h_prev.shape[1])
            output[skip_mask[:, :min_seq]] = self.h_prev[:, :min_seq][skip_mask[:, :min_seq]]
        
        return output

