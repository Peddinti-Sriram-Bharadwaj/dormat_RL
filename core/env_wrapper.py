import gymnasium as gym
import numpy as np
from gymnasium import spaces

class PadEnvWrapper(gym.Wrapper):
    """
    Wraps an environment to pad its observation and action spaces to a fixed maximum size.
    This allows a single network architecture to interact with multiple environments.
    """
    def __init__(self, env: gym.Env, max_state_dim: int = 8, max_action_dim: int = 4):
        super().__init__(env)
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        
        self.original_obs_dim = env.observation_space.shape[0]
        self.original_action_dim = env.action_space.n
        
        # Override observation space
        low = np.full((max_state_dim,), -np.inf, dtype=np.float32)
        high = np.full((max_state_dim,), np.inf, dtype=np.float32)
        
        if hasattr(env.observation_space, 'low'):
            low[:self.original_obs_dim] = env.observation_space.low
        if hasattr(env.observation_space, 'high'):
            high[:self.original_obs_dim] = env.observation_space.high
            
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Override action space
        self.action_space = spaces.Discrete(max_action_dim)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._pad_obs(obs), info
        
    def step(self, action: int):
        # Map the chosen action to a valid action for the underlying environment
        # e.g., if max_action_dim=4, but env only has 2 actions (0, 1)
        valid_action = action % self.original_action_dim
        
        obs, reward, terminated, truncated, info = self.env.step(valid_action)
        return self._pad_obs(obs), reward, terminated, truncated, info
        
    def _pad_obs(self, obs: np.ndarray) -> np.ndarray:
        padded_obs = np.zeros(self.max_state_dim, dtype=np.float32)
        padded_obs[:self.original_obs_dim] = obs
        return padded_obs
