import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
            
        # Output layer
        self.output_layer = nn.Linear(in_dim, output_dim)
        
    def forward(self, x: torch.Tensor, return_preactivations: bool = False):
        """
        Forward pass.
        If return_preactivations is True, returns (output, preactivations)
        where preactivations is a list of tensors containing the pre-activation 
        outputs of each hidden layer.
        """
        preactivations = []
        out = x
        for layer in self.layers:
            out = layer(out)
            if return_preactivations:
                preactivations.append(out)
            out = F.relu(out)
            
        out = self.output_layer(out)
        
        if return_preactivations:
            return out, preactivations
        return out
