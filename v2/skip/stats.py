import json
from dataclasses import dataclass, asdict, field
from typing import List

@dataclass
class SkipStats:
    """Track FLOPs reduction statistics"""
    total_steps: int = 0
    total_layers: int = 0
    sequence_length: int = 0
    
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
        Compute FLOPs ratio: active_tokens * executed_layers / (total_steps * seq_len * total_layers)
        """
        if self.total_steps == 0 or self.sequence_length == 0 or self.total_layers == 0:
            return 1.0
        
        total_flops = sum(
            active * executed 
            for active, executed in zip(
                self.active_tokens_per_step, 
                self.executed_layers_per_step
            )
        )
        
        baseline_flops = self.total_steps * self.sequence_length * self.total_layers
        
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

