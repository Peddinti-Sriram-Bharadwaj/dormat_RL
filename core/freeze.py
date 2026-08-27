import torch
import torch.nn as nn
from typing import List

def freeze_active_neurons(network, dormant_indices: List[torch.Tensor]):
    """
    Freezes the active (non-dormant) neurons by registering backward hooks
    that zero out their gradients.
    
    Args:
        network: The MLP network.
        dormant_indices: A list of boolean tensors indicating dormant neurons for each hidden layer.
                         Active neurons are ~dormant_indices.
    """
    hooks = []
    
    def get_weight_hook(active_mask, dim):
        """
        Returns a hook function that zeroes out gradients along the specified dimension
        for the active neurons.
        dim=0 corresponds to the output dimension (incoming weights of the neuron).
        dim=1 corresponds to the input dimension (outgoing weights of the neuron).
        """
        def hook(grad):
            new_grad = grad.clone()
            if dim == 0:
                new_grad[active_mask, :] = 0.0
            elif dim == 1:
                new_grad[:, active_mask] = 0.0
            return new_grad
        return hook
        
    def get_bias_hook(active_mask):
        def hook(grad):
            new_grad = grad.clone()
            new_grad[active_mask] = 0.0
            return new_grad
        return hook

    for i, (layer, is_dormant) in enumerate(zip(network.layers, dormant_indices)):
        is_active = ~is_dormant
        
        # Freeze incoming weights and biases for active neurons
        h_w = layer.weight.register_hook(get_weight_hook(is_active, dim=0))
        hooks.append(h_w)
        if layer.bias is not None:
            h_b = layer.bias.register_hook(get_bias_hook(is_active))
            hooks.append(h_b)
            
        # Freeze outgoing weights from active neurons
        if i + 1 < len(network.layers):
            next_layers = [network.layers[i + 1]]
        else:
            next_layers = network.output_layers
            
        for next_layer in next_layers:
            h_out = next_layer.weight.register_hook(get_weight_hook(is_active, dim=1))
            hooks.append(h_out)
        
    return hooks