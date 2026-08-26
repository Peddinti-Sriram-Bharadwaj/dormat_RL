import torch
import torch.nn as nn
import math
from typing import List
from .network import MLP

def reset_linear_layer_weights(layer: nn.Linear, indices: torch.Tensor):
    """
    Reinitializes the incoming weights and biases for the specified neurons
    in a Linear layer, matching PyTorch's default initialization.
    """
    if indices.sum() == 0:
        return
        
    with torch.no_grad():
        # Get the original initialization bounds
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        
        # Reinitialize incoming weights
        new_weights = torch.empty((indices.sum().item(), layer.in_features), device=layer.weight.device)
        nn.init.kaiming_uniform_(new_weights, a=math.sqrt(5))
        layer.weight[indices, :] = new_weights
        
        # Reinitialize biases
        if layer.bias is not None:
            new_biases = torch.empty(indices.sum().item(), device=layer.bias.device)
            nn.init.uniform_(new_biases, -bound, bound)
            layer.bias[indices] = new_biases

def recycle_dormant_neurons(network: MLP, dormant_indices: List[torch.Tensor]):
    """
    Applies the ReDo algorithm:
    1. Reinitializes incoming weights of dormant neurons.
    2. Zeroes out outgoing weights of dormant neurons.
    
    Args:
        network: The MLP network.
        dormant_indices: A list of boolean tensors indicating dormant neurons for each hidden layer.
    """
    with torch.no_grad():
        for i, (layer, is_dormant) in enumerate(zip(network.layers, dormant_indices)):
            if is_dormant.sum() == 0:
                continue
                
            # 1. Reinitialize incoming weights
            reset_linear_layer_weights(layer, is_dormant)
            
            # 2. Zero-out outgoing weights
            # Find the next layer
            if i + 1 < len(network.layers):
                next_layer = network.layers[i + 1]
            else:
                next_layer = network.output_layer
                
            # next_layer.weight has shape (out_features, in_features)
            # The dormant neurons in the current layer correspond to the in_features of the next layer
            next_layer.weight[:, is_dormant] = 0.0
