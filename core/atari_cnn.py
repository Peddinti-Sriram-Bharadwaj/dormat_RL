"""
Atari DQN CNN with built-in dormancy tracking.

Architecture: Nature DQN (Mnih et al. 2015)
  Conv1: 8x8 kernel, stride 4, 32 filters
  Conv2: 4x4 kernel, stride 2, 64 filters
  Conv3: 3x3 kernel, stride 1, 64 filters
  FC1:   512 neurons
  FC2:   num_actions (output)

Dormancy for FC neurons: mean post-ReLU activation < tau
Dormancy for Conv filters: mean post-ReLU activation across all spatial
                           positions and batch samples < tau
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class AtariCNN(nn.Module):
    def __init__(self, num_actions: int, in_channels: int = 4):
        """
        Args:
            num_actions: number of discrete actions for the game
            in_channels: number of stacked frames (default 4)
        """
        super().__init__()

        # Convolutional backbone
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute flattened size after conv layers for 84x84 input
        # 84 -> (84-8)/4+1 = 20 -> (20-4)/2+1 = 9 -> (9-3)/1+1 = 7
        # Output: 64 filters × 7 × 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc_out = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        """
        x: (batch, 4, 84, 84) normalized to [0, 1]

        Returns:
            q_values: (batch, num_actions)
            activations (optional): dict with per-layer post-ReLU activations
        """
        a1 = F.relu(self.conv1(x))   # (B, 32, 20, 20)
        a2 = F.relu(self.conv2(a1))  # (B, 64, 9, 9)
        a3 = F.relu(self.conv3(a2))  # (B, 64, 7, 7)

        flat = a3.reshape(a3.size(0), -1)  # (B, 3136)
        a4 = F.relu(self.fc1(flat))        # (B, 512)
        q  = self.fc_out(a4)               # (B, num_actions)

        if return_activations:
            return q, {
                "conv1": a1,   # (B, 32, 20, 20)
                "conv2": a2,   # (B, 64, 9, 9)
                "conv3": a3,   # (B, 64, 7, 7)
                "fc1":   a4,   # (B, 512)
            }
        return q


def compute_dormancy(activations: Dict[str, torch.Tensor],
                     tau: float = 0.025) -> Dict[str, dict]:
    """
    Compute dormancy for each layer.

    For FC layers: a neuron is dormant if its mean activation across the
    batch is < tau (same criterion as the classic control experiments).

    For Conv layers: a filter is dormant if its mean activation across
    ALL spatial positions AND the entire batch is < tau. Each filter
    produces a 2D feature map — we average over (batch, H, W).

    Returns a dict with per-layer stats:
        'n_total':   total number of units (filters or neurons)
        'n_dormant': count of dormant units
        'pct':       dormancy percentage
        'mask':      boolean tensor (True = dormant)
    """
    stats = {}

    for name, act in activations.items():
        if act.dim() == 4:
            # Conv layer: act is (B, C, H, W)
            # Mean over batch and spatial dims → per-filter mean (C,)
            per_filter_mean = act.mean(dim=(0, 2, 3))   # shape: (C,)
            dormant_mask = per_filter_mean < tau
        elif act.dim() == 2:
            # FC layer: act is (B, N)
            per_neuron_mean = act.mean(dim=0)            # shape: (N,)
            dormant_mask = per_neuron_mean < tau
        else:
            continue

        n_total   = dormant_mask.numel()
        n_dormant = dormant_mask.sum().item()
        stats[name] = {
            "n_total":   n_total,
            "n_dormant": int(n_dormant),
            "pct":       100.0 * n_dormant / n_total,
            "mask":      dormant_mask,
            "mean_acts": per_filter_mean if act.dim() == 4 else per_neuron_mean,
        }

    return stats
