"""
Experiment 12: Width-Varied Speed Comparison Grid

Sweeps over widths [64, 128, 256, 512] and all 6 game permutations
(CartPole, Acrobot, MountainCar w/ reward shaping) to measure whether
recycled dormant neurons learn Task B faster or slower than a fresh network.

Metric: Steps to convergence (lower = faster = better positive transfer).
"""

import gymnasium as gym
import torch
import numpy as np
import itertools
from collections import deque
import multiprocessing as mp

from agents.multihead_agent import MultiHeadDQNAgent
from core.env_wrapper import PadEnvWrapper


class ShapedMountainCarWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    def reward(self, reward):
        velocity = self.env.unwrapped.state[1]
        return reward + 100.0 * abs(velocity)


def get_threshold(env_id):
    if "CartPole" in env_id:
        return 400.0
    elif "Acrobot" in env_id:
        return -100.0
    elif "MountainCar" in env_id:
        return 0.0   # shaped reward threshold
    return 100.0


def make_env(env_id):
    env = gym.make(env_id)
    if "MountainCar" in env_id:
        env = ShapedMountainCarWrapper(env)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
    return env


def train_to_convergence(agent, env_id, max_steps, threshold, phase_name=""):
    env = make_env(env_id)
    state, _ = env.reset()

    epsilon_start = 1.0 if phase_name in ("Phase 1", "Scratch") else 0.5
    epsilon_end = 0.05
    epsilon_decay = max_steps // 5

    episode_reward = 0
    recent_rewards = deque(maxlen=10)

    for step in range(1, max_steps + 1):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step / epsilon_decay)
        action = agent.select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if done:
            recent_rewards.append(episode_reward)
            state, _ = env.reset()
            episode_reward = 0

            if len(recent_rewards) >= 10 and np.mean(recent_rewards) >= threshold:
                env.close()
                return step

    env.close()
    return max_steps  # Did not converge (DNF)


def run_single_seed(seed, width, env_a, env_b):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    thresh_a = get_threshold(env_a)
    thresh_b = get_threshold(env_b)

    # --- Recycled network ---
    agent = MultiHeadDQNAgent(
        state_dim=8, action_dim=4, hidden_dims=[width, width],
        replay_ratio=0.25, device="cpu", num_heads=2
    )
    agent.set_head(0)
    train_to_convergence(agent, env_a, 150000, thresh_a, "Phase 1")
    agent.freeze_active_neurons_and_reset_dormant(0.025)
    agent.set_head(1)
    agent.memory.buffer.clear()
    steps_recycled = train_to_convergence(agent, env_b, 150000, thresh_b, "Phase 2")

    # --- Scratch network ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    scratch = MultiHeadDQNAgent(
        state_dim=8, action_dim=4, hidden_dims=[width, width],
        replay_ratio=0.25, device="cpu", num_heads=1
    )
    scratch.set_head(0)
    steps_scratch = train_to_convergence(scratch, env_b, 150000, thresh_b, "Scratch")

    print(f"  [W={width}|{env_a[:4]}->{env_b[:4]}|S{seed}] Recycled={steps_recycled} Scratch={steps_scratch}")
    return steps_recycled, steps_scratch


def run_experiment_12(widths, num_seeds=5, num_cores=6):
    games = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    pairs = list(itertools.permutations(games, 2))

    # Build all task args
    tasks = []
    task_labels = []
    for width in widths:
        for env_a, env_b in pairs:
            for seed in range(num_seeds):
                tasks.append((seed, width, env_a, env_b))
            task_labels.append((width, env_a, env_b))

    print(f"\n{'='*60}")
    print(f"Experiment 12: Width-Varied Speed Grid")
    print(f"Widths: {widths} | Pairs: {len(pairs)} | Seeds: {num_seeds}")
    print(f"Total tasks: {len(tasks)} | Cores: {num_cores}")
    print(f"{'='*60}")

    with mp.Pool(num_cores) as pool:
        raw = pool.starmap(run_single_seed, tasks)

    # Aggregate
    rows = []
    idx = 0
    for width in widths:
        for env_a, env_b in pairs:
            recycled = [raw[idx + s][0] for s in range(num_seeds)]
            scratch  = [raw[idx + s][1] for s in range(num_seeds)]
            idx += num_seeds

            r_mean, r_std = np.mean(recycled), np.std(recycled)
            s_mean, s_std = np.mean(scratch),  np.std(scratch)
            delta = s_mean - r_mean  # positive = recycled faster

            rows.append((width, env_a, env_b, r_mean, r_std, s_mean, s_std, delta))
            print(f"W={width} | {env_a[:4]}->{env_b[:4]} | Rec={r_mean:.0f}±{r_std:.0f} | Scr={s_mean:.0f}±{s_std:.0f} | Δ={delta:.0f}")

    # Write markdown table
    with open("exp12_results.md", "w") as f:
        f.write("## Experiment 12: Width-Varied Speed Comparison Grid\n")
        f.write("**Date:** 2026-08-27 | **Seeds:** 5 | **MountainCar:** velocity reward shaping\n\n")
        f.write("| Width | Task A | Task B | Recycled Steps (Mean±Std) | Scratch Steps (Mean±Std) | Δ Steps |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            w, ea, eb, rm, rs, sm, ss, d = r
            dnf = "(DNF)" if rm >= 149999 else ""
            f.write(f"| {w} | {ea} | {eb} | `{rm:.0f} ± {rs:.0f}` {dnf} | `{sm:.0f} ± {ss:.0f}` | **{d:+.0f}** |\n")

    print(f"\nResults written to exp12_results.md")


def main():
    mp.set_start_method('spawn', force=True)
    run_experiment_12(widths=[64, 128, 256, 512], num_seeds=5, num_cores=6)


if __name__ == "__main__":
    main()
