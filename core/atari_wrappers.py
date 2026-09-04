"""
Atari Preprocessing Wrappers

Standard preprocessing pipeline from Mnih et al. 2015:
1. Frame skip (repeat action for 4 frames, take max of last 2 to remove flicker)
2. Grayscale + resize to 84x84
3. Frame stacking (4 consecutive frames as input)
"""

import numpy as np
import cv2
import gymnasium as gym
from collections import deque


class NoopResetWrapper(gym.Wrapper):
    """Take a random number of no-ops on reset to increase state diversity."""
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipWrapper(gym.Wrapper):
    """Return max over last 2 frames; repeat action for skip frames."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = deque(maxlen=2)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class GrayscaleResizeWrapper(gym.ObservationWrapper):
    """Convert to grayscale and resize to 84x84."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]


class ClipRewardWrapper(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1}."""
    def reward(self, reward):
        return np.sign(reward)


class FrameStackWrapper(gym.Wrapper):
    """Stack the last n_frames observations along the channel dimension."""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self._frames = deque(maxlen=n_frames)
        shape = env.observation_space.shape[:2]  # (84, 84)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(n_frames, *shape), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self._frames.append(obs[:, :, 0])
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs[:, :, 0])
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self._frames, axis=0)  # (n_frames, 84, 84)


def make_atari_env(game_id: str, clip_rewards: bool = True):
    """
    Create a preprocessed Atari environment.
    game_id: e.g., 'ALE/Pong-v5', 'ALE/Breakout-v5'
    Returns env with obs shape (4, 84, 84) and clipped rewards.
    """
    import ale_py
    import gymnasium
    gymnasium.register_envs(ale_py)

    env = gymnasium.make(game_id, render_mode=None)
    env = NoopResetWrapper(env, noop_max=30)
    env = MaxAndSkipWrapper(env, skip=4)
    env = GrayscaleResizeWrapper(env)
    if clip_rewards:
        env = ClipRewardWrapper(env)
    env = FrameStackWrapper(env, n_frames=4)
    return env
