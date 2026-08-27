import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List

from agents.dqn_agent import DQNAgent
from core.multihead_network import MultiHeadMLP
from core.dormancy import calculate_dormancy_scores
from core.multihead_freeze import multihead_freeze_active_neurons

class MultiHeadDQNAgent(DQNAgent):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256], num_heads: int = 2, **kwargs):
        # Call super first to initialize everything else
        super().__init__(state_dim, action_dim, hidden_dims, **kwargs)
        
        self.current_head = 0
        
        # Override the networks with MultiHeadMLP
        self.network = MultiHeadMLP(state_dim, action_dim, hidden_dims, num_heads=num_heads).to(self.device)
        self.target_network = MultiHeadMLP(state_dim, action_dim, hidden_dims, num_heads=num_heads).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Reinitialize optimizer for the new network
        self.optimizer = optim.Adam(self.network.parameters(), lr=kwargs.get("lr", 1e-3))
        
    def set_head(self, head_idx: int):
        self.current_head = head_idx

    def select_action(self, state: np.ndarray, epsilon: float = 0.0, head_idx: int = None) -> int:
        if head_idx is None:
            head_idx = self.current_head
            
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
            
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.network(state_t, head_idx=head_idx)
            return q_values.argmax(dim=1).item()

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
        q_values = self.network(states, head_idx=self.current_head).gather(1, actions)
        
        # Max Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states, head_idx=self.current_head).max(1, keepdim=True)[0]
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def evaluate_dormancy(self, dormancy_tau: float = 0.025, redo: bool = False) -> List[float]:
        if len(self.memory) < self.batch_size:
            return [0.0] * len(self.network.layers)
            
        states, _, _, _, _ = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        
        with torch.no_grad():
            _, activations = self.network(states, return_activations=True, head_idx=self.current_head)
            
        dormant_indices, percentages = calculate_dormancy_scores(activations, dormancy_tau)
        
        # Skipping redo logic for brevity since it's rarely used directly here
        return percentages

    def freeze_active_neurons_and_reset_dormant(self, dormancy_tau: float = 0.025):
        if len(self.memory) < self.batch_size:
            print("Not enough data to evaluate dormancy.")
            return

        states, _, _, _, _ = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        
        with torch.no_grad():
            _, activations = self.network(states, return_activations=True, head_idx=self.current_head)
            
        dormant_indices, percentages = calculate_dormancy_scores(activations, dormancy_tau)
        
        # Store the active masks (nA) for use in masked evaluation (Experiment 11)
        self.active_masks = [~is_dormant for is_dormant in dormant_indices]
        
        from core.redo import reset_linear_layer_weights
        with torch.no_grad():
            for layer, is_dormant in zip(self.network.layers, dormant_indices):
                reset_linear_layer_weights(layer, is_dormant)
                
        self.hooks = multihead_freeze_active_neurons(self.network, dormant_indices)
        
        self.target_network.load_state_dict(self.network.state_dict())
        
        return percentages

    def evaluate_masked(self, env_id: str, head_idx: int = 0, n_episodes: int = 10) -> float:
        """
        Evaluates Task A performance using ONLY nA neurons.
        All nB (recycled dormant) neuron activations are zeroed out
        at every hidden layer during the forward pass.
        """
        import gymnasium as gym
        from core.env_wrapper import PadEnvWrapper

        assert hasattr(self, 'active_masks'), "Must call freeze_active_neurons_and_reset_dormant first."

        env = gym.make(env_id)
        env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
        total_reward = 0.0

        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.network.forward_masked(state_t, self.active_masks, head_idx=head_idx)
                action = q_values.argmax(dim=1).item()
                state, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                done = terminated or truncated
            total_reward += ep_reward

        env.close()
        return total_reward / n_episodes
