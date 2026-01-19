import torch
import torch.nn.functional as F

class LayerSkipPolicy:
    """Layer-level skipping within a denoising step"""
    
    def __init__(self, tau_layer=0.99, enabled=True):
        self.tau_layer = tau_layer
        self.enabled = enabled
        self.x_prev_in = None  # Previous layer input
        self.executed_layers = []  # Track which layers executed vs skipped
        
    def reset(self):
        """Reset state between forward passes"""
        self.x_prev_in = None
        self.executed_layers = []
    
    def should_skip_layer(self, x_in):
        """
        Decide if current layer should be skipped
        Returns: (should_skip: bool, similarity: float)
        """
        if self.x_prev_in is None or not self.enabled:
            self.x_prev_in = x_in.clone() if isinstance(x_in, torch.Tensor) else x_in
            return False, 0.0
        
        # Ensure same shape
        if x_in.shape != self.x_prev_in.shape:
            self.x_prev_in = x_in.clone() if isinstance(x_in, torch.Tensor) else x_in
            return False, 0.0
        
        # Compute mean cosine similarity across tokens
        x_in_norm = F.normalize(x_in, p=2, dim=-1)  # [B, S, H]
        x_prev_norm = F.normalize(self.x_prev_in, p=2, dim=-1)
        
        # Mean cosine similarity per token, then average
        cos_sim_per_token = (x_in_norm * x_prev_norm).sum(dim=-1)  # [B, S]
        mean_sim = cos_sim_per_token.mean().item()
        
        should_skip = mean_sim >= self.tau_layer
        
        # Update x_prev_in for next layer
        self.x_prev_in = x_in.clone() if isinstance(x_in, torch.Tensor) else x_in
        
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

