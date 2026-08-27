import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MultiHeadMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], num_heads: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
            
        # Output layers (multi-head support)
        self.output_layers = nn.ModuleList([nn.Linear(in_dim, output_dim) for _ in range(num_heads)])
        
    def forward(self, x: torch.Tensor, return_activations: bool = False, head_idx: int = 0):
        """
        Forward pass.
        If return_activations is True, returns (output, activations)
        where activations is a list of tensors containing the post-activation 
        outputs of each hidden layer.
        """
        activations = []
        out = x
        for layer in self.layers:
            out = layer(out)
            out = F.relu(out)
            if return_activations:
                activations.append(out)
            
        out = self.output_layers[head_idx](out)
        
        if return_activations:
            return out, activations
        return out
