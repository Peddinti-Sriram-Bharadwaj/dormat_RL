import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, Dict, Any, List

from core.network import MLP
from core.dormancy import calculate_dormancy_scores
from core.redo import recycle_dormant_neurons

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: List[int] = [256, 256],
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005, # soft update for target net
        buffer_size: int = 100000,
        batch_size: int = 64,
        replay_ratio: float = 0.25, # Updates per env step
        device: str = "cpu",
        **kwargs
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.replay_ratio = replay_ratio
        self.device = torch.device(device)
        
        self.network = MLP(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = MLP(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
        # We track how many updates we owe based on the replay ratio
        self.updates_owed = 0.0

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
            
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.network(state_t)
            return q_values.argmax(dim=1).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.updates_owed += self.replay_ratio
        
        updates_to_do = int(self.updates_owed)
        if updates_to_do > 0:
            self.updates_owed -= updates_to_do
            for _ in range(updates_to_do):
                self._update()

    def _update(self):
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Q(s, a)
        q_values = self.network(states).gather(1, actions)
        
        # Max Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def evaluate_dormancy(self, dormancy_tau: float = 0.025, redo: bool = False) -> List[float]:
        """
        Evaluates dormancy on a sampled batch. Optionally applies ReDo.
        Returns the percentage of dormant neurons per layer.
        """
        if len(self.memory) < self.batch_size:
            return [0.0] * len(self.network.layers)
            
        states, _, _, _, _ = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        
        with torch.no_grad():
            _, preactivations = self.network(states, return_preactivations=True)
            
        dormant_indices, percentages = calculate_dormancy_scores(preactivations, dormancy_tau)
        
        if redo:
            recycle_dormant_neurons(self.network, dormant_indices)
            # Re-sync optimizer momentum if using Adam (optional, but good practice)
            # For simplicity in this skeleton, we just reinitialize the optimizer to clear momentum for recycled weights
            # Alternatively, we could manually zero out momentum buffers for the specific neurons.
            # To keep the skeleton simple, we just leave it as is or reset the optimizer state.
            
            # Update target network to match the newly recycled weights to prevent large TD spikes
            # But the paper says target network can be updated gradually or immediately. 
            # We will just do a hard copy for the affected weights, but simplest is to just copy the whole layer if we reset.
            # We'll just let the target network follow via soft updates.
            
        return percentages
