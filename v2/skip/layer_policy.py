import torch
import torch.nn.functional as F

class LayerSkipPolicy:
    """Layer-level skipping within a denoising step"""
    
    def __init__(self, tau_layer=0.99, enabled=True):
        self.tau_layer = tau_layer
        self.enabled = enabled
        self.x_prev_in = None  # Previous layer's INPUT (not output) - for input-to-input comparison
        self.executed_layers = []  # Track which layers executed vs skipped
        self.consecutive_skips = 0  # Track consecutive skips to prevent cascade
        
    def reset(self):
        """Reset state between forward passes"""
        self.x_prev_in = None
        self.executed_layers = []
        self.consecutive_skips = 0
    
    def should_skip_layer(self, x_in):
        """
        Decide if current layer should be skipped by comparing input-to-input.
        
        Compares Layer L's INPUT to Layer L-1's INPUT (stored in x_prev_in).
        This matches the task requirement: "Compute the cosine similarity of the 
        input hidden states between adjacent layers within the same denoising step."
        
        Args:
            x_in: Current layer's input hidden states [B, S, H]
            
        Returns:
            (should_skip: bool, similarity: float)
        """
        if self.x_prev_in is None or not self.enabled:
            # First layer: no previous input to compare, so don't skip
            # x_prev_in will be updated in forward_with_skip after processing
            return False, 0.0
        
        # Ensure same shape
        if x_in.shape != self.x_prev_in.shape:
            # Shape mismatch: don't skip, x_prev_in will be updated in forward_with_skip
            return False, 0.0
        
        # Compute mean cosine similarity across tokens
        # Compare current layer's INPUT to previous layer's INPUT
        x_in_norm = F.normalize(x_in, p=2, dim=-1)  # [B, S, H]
        x_prev_norm = F.normalize(self.x_prev_in, p=2, dim=-1)
        
        # Mean cosine similarity per token, then average
        cos_sim_per_token = (x_in_norm * x_prev_norm).sum(dim=-1)  # [B, S]
        mean_sim = cos_sim_per_token.mean().item()
        
        should_skip = mean_sim >= self.tau_layer
        
        # NOTE: x_prev_in is updated in forward_with_skip() to the current layer's INPUT
        # (not output) after processing, ensuring we always compare input-to-input
        
        return should_skip, mean_sim
    
    def record_layer_execution(self, layer_idx, executed):
        """Record whether layer was executed or skipped"""
        self.executed_layers.append((layer_idx, executed))
    
    def get_executed_count(self):
        """Get count of executed layers"""
        return sum(1 for _, executed in self.executed_layers if executed)
    
    def get_total_layers(self):
        """Get total number of layers processed"""
        return len(self.executed_layers)

