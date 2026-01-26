import json
from dataclasses import dataclass, asdict, field
from typing import List

@dataclass
class SkipStats:
    """Track FLOPs reduction statistics"""
    total_steps: int = 0
    total_layers: int = 0
    sequence_length: int = 0
    batch_size: int = 1  # NEW: Track batch size for correct FLOPs normalization
    
    # Per-step stats
    active_tokens_per_step: List[int] = field(default_factory=list)
    executed_layers_per_step: List[int] = field(default_factory=list)
    
    # Thresholds used
    token_tau: float = 0.0
    layer_tau: float = 0.0
    
    def record_step(self, active_tokens: int, executed_layers: int):
        """Record stats for one denoising step"""
        self.active_tokens_per_step.append(active_tokens)
        self.executed_layers_per_step.append(executed_layers)
        self.total_steps += 1
    
    def compute_flops_ratio(self):
        """
        Compute FLOPs ratio accounting for token and/or layer skipping.
        
        Three cases based on what's being skipped:
        
        1. Token skip only (variable active_tokens, constant executed_layers=L_total):
           FLOPs = Σ(A × L_total) / (T × S × L_total × B) = Σ(A) / (T × S × B)
           
        2. Layer skip only (constant active_tokens=S×B, variable executed_layers):
           FLOPs = (T × S × Σ(L_exec)/T × B) / (T × S × L_total × B) = Σ(L_exec) / (T × L_total)
           
        3. Both (variable A and variable L_exec):
           FLOPs = Σ(A × L_exec) / (T × S × L_total × B)
        
        Where:
        - A = active_tokens (tokens NOT skipped by token policy) per step
        - L_exec = executed_layers (layers NOT skipped by layer policy) per step
        - L_total = total_layers
        - T = total_steps
        - S = sequence_length (block_size)
        - B = batch_size
        """
        if self.total_steps == 0 or self.sequence_length == 0 or self.total_layers == 0:
            return 1.0
        
        effective_batch_size = max(1, self.batch_size)
        
        # Detect which type of skipping is active
        active_is_variable = len(set(self.active_tokens_per_step)) > 1
        layers_is_variable = len(set(self.executed_layers_per_step)) > 1
        
        # Calculate total FLOPs based on skipping configuration
        if active_is_variable and layers_is_variable:
            # Case 3: BOTH token and layer skipping
            # FLOPs = Σ(A × L_exec)
            total_flops = sum(a * l for a, l in zip(self.active_tokens_per_step, 
                                                      self.executed_layers_per_step))
        elif active_is_variable and not layers_is_variable:
            # Case 1: TOKEN SKIP ONLY (all layers executed: L_exec = L_total)
            # FLOPs = Σ(A) × L_total
            total_active = sum(self.active_tokens_per_step)
            total_flops = total_active * self.total_layers
        elif not active_is_variable and layers_is_variable:
            # Case 2: LAYER SKIP ONLY (all tokens active: A = S × B)
            # FLOPs = T × S × B × Σ(L_exec) / T = S × B × Σ(L_exec)
            # But we normalize per-step, so: Σ(L_exec) only
            total_layers_executed = sum(self.executed_layers_per_step)
            # Scale to per-step average: (T * S * B * avg(L_exec)) / (T * S * L_total * B)
            total_flops = (self.total_steps * self.sequence_length * 
                          total_layers_executed / self.total_steps * effective_batch_size)
        else:
            # Case 0: NO SKIPPING (all constant)
            # A = S × B for all steps, L_exec = L_total for all steps
            total_flops = (self.total_steps * self.sequence_length * 
                          self.total_layers * effective_batch_size)
        
        # Baseline FLOPs: T × S × L × B (all tokens, all layers, all batch items)
        baseline_flops = (self.total_steps * self.sequence_length * 
                         self.total_layers * effective_batch_size)
        
        if baseline_flops == 0:
            return 1.0
        
        return total_flops / baseline_flops
    
    def compute_flops_reduction(self):
        """Compute FLOPs reduction percentage"""
        return 1.0 - self.compute_flops_ratio()
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            **asdict(self),
            'flops_ratio': self.compute_flops_ratio(),
            'flops_reduction': self.compute_flops_reduction()
        }
    
    def save_jsonl(self, filepath):
        """Append stats to JSONL file"""
        import os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'a') as f:
            f.write(json.dumps(self.to_dict()) + '\n')

