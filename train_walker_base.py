"""
Phase 1 of the MuJoCo remote-control pipeline: train the low-level
"remote control" base policy -- a SAC agent that learns fine-grained,
low-jitter joint control on Walker2d-v5 (forward locomotion).

This is intentionally the bare minimum: standard SAC, standard reward
(forward velocity + healthy bonus - control cost, all built into the
Gymnasium env already). No macro-command interface yet -- that's the
next phase, once this base policy walks smoothly.
"""

import os

import gymnasium as gym
import torch
import numpy as np
import random

from core.sac_agent import SACAgent

CHECKPOINT_PATH = "agents/walker_base_actor.pt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(env_name="Walker2d-v5", seed=0, max_steps=1_000_000, warmup_steps=5_000, eval_every=20_000):
    set_seed(seed)
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    agent = SACAgent(state_dim, action_dim, action_scale=action_scale, device="auto")
    print(f"  Using device: {agent.device}")

    state, _ = env.reset(seed=seed)
    ep_ret, ep_len, ep_returns = 0.0, 0, []

    for step in range(1, max_steps + 1):
        if step < warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)

        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        agent.step(state, action, reward, next_state, float(term))
        state, ep_ret, ep_len = next_state, ep_ret + reward, ep_len + 1

        if done:
            ep_returns.append(ep_ret)
            state, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        if step % eval_every == 0:
            mean_r = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            eval_r, eval_jitter = evaluate(agent, env_name)
            print(f"  step {step:>7d} | episodes={len(ep_returns):>5d} | train mean20={mean_r:7.1f} "
                  f"| eval_return={eval_r:7.1f} | eval_action_jitter={eval_jitter:.4f}")
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save(agent.actor.state_dict(), CHECKPOINT_PATH)

    env.close()
    return agent


def evaluate(agent, env_name, n_episodes=3, max_ep_steps=1000):
    """Deterministic rollout; jitter = mean |a_t - a_{t-1}| across actuators."""
    env = gym.make(env_name)
    returns, jitters = [], []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=9000 + ep)
        ep_ret, prev_action, deltas, t = 0.0, None, [], 0
        while t < max_ep_steps:
            action = agent.select_action(state, deterministic=True)
            if prev_action is not None:
                deltas.append(np.mean(np.abs(action - prev_action)))
            prev_action = action
            state, r, term, trunc, _ = env.step(action)
            ep_ret += r
            t += 1
            if term or trunc:
                break
        returns.append(ep_ret)
        jitters.append(np.mean(deltas) if deltas else 0.0)
    env.close()
    return float(np.mean(returns)), float(np.mean(jitters))


if __name__ == "__main__":
    print("[Phase 1] Training base Walker2d SAC policy (the 'remote control' body)...")
    agent = train()
    print("\n[Final Evaluation]")
    final_r, final_jitter = evaluate(agent, "Walker2d-v5", n_episodes=10)
    print(f"  mean return over 10 episodes = {final_r:.1f} | mean action jitter = {final_jitter:.4f}")
