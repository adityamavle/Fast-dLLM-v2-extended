import torch
import torch.nn.functional as F

class TokenSkipPolicy:
    """Token-level skipping across denoising steps"""
    
    def __init__(self, tau_token=0.99, enabled=True, probe_layer_idx=0):
        self.tau_token = tau_token
        self.enabled = enabled
        self.probe_layer_idx = probe_layer_idx
        self.h_prev = None  # Previous step probe hidden states [B, S, H] (after first layer)
        self.h_prev_final = None  # Previous step final hidden states [B, S, H] (after all layers)
        
    def reset(self):
        """Reset state between sequences"""
        self.h_prev = None
        self.h_prev_final = None
    
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
        if self.h_prev is not None and self.h_prev.shape == (B, S, H):
            output = self.h_prev.clone()
        else:
            output = torch.zeros(B, S, H, device=device, dtype=packed_hidden.dtype)
        
        # Scatter active tokens (overwrite skipped tokens where active)
        if len(active_indices) > 0:
            batch_idx = active_indices[:, 0]
            seq_idx = active_indices[:, 1]
            output[batch_idx, seq_idx] = packed_hidden
        
        return output
    
    def merge_skipped_tokens(self, h_current, skip_mask, h_prev_final=None):
        """
        Merge current hidden states with previous hidden states for skipped tokens.
        This is called after the full forward pass to replace skipped tokens.
        
        Args:
            h_current: Current hidden states [B, S, H] (after full forward pass)
            skip_mask: Boolean mask [B, S], True = skip
            h_prev_final: Previous step's final hidden states [B, S, H] (optional, uses h_prev if None)
            
        Returns:
            merged_hidden: [B, S, H] with skipped tokens replaced
        """
        if not self.enabled or skip_mask is None or not skip_mask.any():
            return h_current
        
        output = h_current.clone()
        
        # Use provided h_prev_final or fall back to h_prev (probe from previous step)
        h_prev_to_use = h_prev_final if h_prev_final is not None else self.h_prev
        
        if h_prev_to_use is not None:
            B, S, H = h_current.shape
            min_seq = min(S, h_prev_to_use.shape[1])
            
            # Replace skipped tokens with previous hidden states
            # Only replace where skip_mask is True
            output[:, :min_seq][skip_mask[:, :min_seq]] = h_prev_to_use[:, :min_seq][skip_mask[:, :min_seq]]
        
        return output
    
    def apply_token_skipping_to_logits(self, model, logits, skip_mask):
        """
        For skipped tokens, replace logits with logits computed from previous hidden states.
        This implements the "reuse previous outputs" behavior.
        
        Args:
            model: The model instance (to compute logits from hidden states)
            logits: Current logits [B, S, V]
            skip_mask: Boolean mask [B, S], True = skip
            
        Returns:
            logits: [B, S, V] with skipped token logits replaced
        """
        if not self.enabled or skip_mask is None or not skip_mask.any():
            return logits
        
        # If we have previous final hidden states, compute logits from them for skipped tokens
        if self.h_prev_final is not None:
            try:
                # Get the language model head
                if hasattr(model, 'lm_head'):
                    lm_head = model.lm_head
                elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                    # For models with tied embeddings, use embed_tokens weight
                    lm_head_weight = model.model.embed_tokens.weight
                else:
                    return logits
                
                # Compute logits from previous hidden states for skipped tokens
                B, S, V = logits.shape
                min_seq = min(S, self.h_prev_final.shape[1])
                
                # Get previous logits for skipped positions
                if hasattr(model, 'lm_head'):
                    prev_logits = lm_head(self.h_prev_final[:, :min_seq])  # [B, min_seq, V]
                else:
                    prev_logits = torch.nn.functional.linear(
                        self.h_prev_final[:, :min_seq], lm_head_weight
                    )
                
                # Replace logits for skipped tokens
                output_logits = logits.clone()
                output_logits[:, :min_seq][skip_mask[:, :min_seq]] = prev_logits[skip_mask[:, :min_seq]]
                
                return output_logits
            except Exception:
                # If anything fails, return original logits
                return logits
        
        return logits

