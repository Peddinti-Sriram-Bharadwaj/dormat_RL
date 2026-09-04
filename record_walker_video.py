"""
Record a video of the trained Walker2d base SAC policy.

Loads the actor checkpoint saved by train_walker_base.py and rolls out
a single deterministic episode, writing frames to an mp4.
"""

import os

os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "/usr/bin/ffmpeg")

import gymnasium as gym
import imageio
import torch

from core.sac_agent import GaussianActor

CHECKPOINT_PATH = "agents/walker_base_actor.pt"
OUTPUT_PATH = "videos/walker_base.mp4"
ENV_NAME = "Walker2d-v5"


def main(max_steps=1000, seed=9000):
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    actor = GaussianActor(state_dim, action_dim, hidden_dims=[256, 256], action_scale=action_scale)
    actor.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    actor.eval()

    state, _ = env.reset(seed=seed)
    frames = [env.render()]
    ep_ret, t = 0.0, 0
    with torch.no_grad():
        while t < max_steps:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            _, _, action = actor.sample(state_t)
            action = action.squeeze(0).numpy()
            state, reward, term, trunc, _ = env.step(action)
            ep_ret += reward
            t += 1
            frames.append(env.render())
            if term or trunc:
                break
    env.close()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    imageio.mimwrite(OUTPUT_PATH, frames, fps=30, quality=8)
    print(f"Saved {len(frames)} frames ({t} steps, return={ep_ret:.1f}) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
