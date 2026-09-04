"""
Minimal Soft Actor-Critic (SAC) for continuous control, used as the
low-level "remote control" base policy on MuJoCo tasks.

Kept structurally close to the repo's existing MLP/DQNAgent conventions:
the actor exposes `return_activations=True` so later circuit-discovery /
neuron-clamping work (as done on the CartPole DQN) can reuse the same
approach on a continuous-action policy.

Reference: Haarnoja et al., 2018 (Soft Actor-Critic, with automatic
temperature tuning).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List

LOG_STD_MIN, LOG_STD_MAX = -20, 2


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        s, a, r, ns, d = zip(*random.sample(self.buffer, batch_size))
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(ns), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy. Post-ReLU hidden activations are
    exposed for later interpretability work, mirroring core/network.py."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int], action_scale: float = 1.0):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()
        in_dim = state_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(in_dim, h))
            in_dim = h
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)
        self.action_scale = action_scale

    def forward(self, state, return_activations: bool = False):
        activations = []
        out = state
        for layer in self.layers:
            out = F.relu(layer(out))
            if return_activations:
                activations.append(out)
        mean = self.mean_head(out)
        log_std = torch.clamp(self.log_std_head(out), LOG_STD_MIN, LOG_STD_MAX)
        if return_activations:
            return mean, log_std, activations
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale
        log_prob = normal.log_prob(x) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean) * self.action_scale


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers, in_dim = [], state_dim + action_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        hidden_dims: List[int] = [256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 200_000,
        batch_size: int = 256,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.gamma, self.tau, self.batch_size = gamma, tau, batch_size
        self.action_dim = action_dim

        self.actor = GaussianActor(state_dim, action_dim, hidden_dims, action_scale).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optim = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        # automatic entropy temperature tuning
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

        self.memory = ReplayBuffer(buffer_size)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic: bool = False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor.sample(state_t)
            else:
                action, _, _ = self.actor.sample(state_t)
        return action.squeeze(0).cpu().numpy()

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) >= self.batch_size:
            self._update()

    def _update(self):
        s, a, r, ns, d = self.memory.sample(self.batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_a, next_log_prob, _ = self.actor.sample(ns)
            q1_next = self.q1_target(ns, next_a)
            q2_next = self.q2_target(ns, next_a)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = r + (1 - d) * self.gamma * q_next

        q1_pred = self.q1(s, a)
        q2_pred = self.q2(s, a)
        q_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        new_a, log_prob, _ = self.actor.sample(s)
        q1_new = self.q1(s, new_a)
        q2_new = self.q2(s, new_a)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        for target, source in [(self.q1_target, self.q1), (self.q2_target, self.q2)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
