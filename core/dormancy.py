import torch
from typing import List, Tuple, Dict

def calculate_dormancy_scores(activations: List[torch.Tensor], tau: float = 0.025) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Calculates the dormancy scores for each neuron in each layer.
    
    Args:
        activations: A list of tensors, each of shape (batch_size, hidden_dim),
            representing the post-activation outputs for each layer.
        tau: The threshold below which a neuron is considered dormant.
        
    Returns:
        dormant_indices: A list of boolean tensors, one for each layer, where True
            indicates the neuron is dormant.
        dormant_percentages: A list of floats representing the percentage of dormant
            neurons in each layer.
    """
    dormant_indices = []
    dormant_percentages = []
    
    for layer_acts in activations:
        # layer_acts is (batch_size, hidden_dim)
        # Calculate mean absolute activation per neuron: E_x |h_i(x)|
        mean_abs_acts = torch.mean(torch.abs(layer_acts), dim=0) # Shape: (hidden_dim,)
        
        # Calculate the average of these over all neurons in the layer
        avg_layer_act = torch.mean(mean_abs_acts) # Scalar
        
        # Calculate score s_i
        # Avoid division by zero by adding a small epsilon
        scores = mean_abs_acts / (avg_layer_act + 1e-8)
        
        # Identify dormant neurons
        is_dormant = scores <= tau
        dormant_indices.append(is_dormant)
        
        # Calculate percentage
        percentage = (is_dormant.sum().item() / is_dormant.numel()) * 100.0
        dormant_percentages.append(percentage)
        
    return dormant_indices, dormant_percentages
