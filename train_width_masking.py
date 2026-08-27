"""
Experiment 13: Width-Varied Activation Masking (nA-Only Evaluation)

Sweeps over widths [64, 128, 256, 512] on the fixed CartPole → Acrobot pair.
For each width, measures:
  - % dormant neurons after Phase 1 (size of nB)
  - Task A baseline: all neurons vs nA-only masked
  - Task A final (post Phase 2): all neurons vs nA-only masked

This tells us if the nA-retention effect and structural bias scale with width.
"""

import gymnasium as gym
import torch
import numpy as np
from collections import deque
import multiprocessing as mp

from agents.multihead_agent import MultiHeadDQNAgent
from core.env_wrapper import PadEnvWrapper
from core.dormancy import calculate_dormancy_scores


def get_threshold(env_id):
    if "CartPole" in env_id:
        return 400.0
    elif "Acrobot" in env_id:
        return -100.0
    return 100.0


def train_to_convergence(agent, env_id, max_steps, threshold, phase_name=""):
    env = gym.make(env_id)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
    state, _ = env.reset()

    epsilon_start = 1.0 if phase_name == "Phase 1" else 0.5
    epsilon_end = 0.05
    epsilon_decay = max_steps // 5

    episode_reward = 0
    recent_rewards = deque(maxlen=10)
    conv_step = max_steps
    conv_ep = 0
    ep_count = 0

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
            ep_count += 1
            episode_reward = 0

            if len(recent_rewards) >= 10 and np.mean(recent_rewards) >= threshold:
                conv_step = step
                conv_ep = ep_count
                break

    env.close()
    return conv_step, conv_ep


def evaluate_standard(agent, env_id, head_idx=0, n_episodes=10):
    env = gym.make(env_id)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
    total = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_values = agent.network(state_t, head_idx=head_idx)
            action = q_values.argmax(dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            ep_r += reward
            done = terminated or truncated
        total += ep_r
    env.close()
    return total / n_episodes


def run_single_seed(seed, width, env_a, env_b):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = MultiHeadDQNAgent(
        state_dim=8, action_dim=4, hidden_dims=[width, width],
        replay_ratio=0.25, device="cpu", num_heads=2
    )

    # Phase 1
    agent.set_head(0)
    p1_steps, p1_eps = train_to_convergence(agent, env_a, 150000, get_threshold(env_a), "Phase 1")

    # Baseline eval: all neurons
    baseline_all = evaluate_standard(agent, env_a, head_idx=0)

    # Identify nA masks WITHOUT resetting anything yet
    states_sample, _, _, _, _ = agent.memory.sample(agent.batch_size)
    states_t = torch.FloatTensor(states_sample)
    with torch.no_grad():
        _, activations = agent.network(states_t, return_activations=True, head_idx=0)
    dormant_indices, dormancy_pcts = calculate_dormancy_scores(activations, 0.025)
    agent.active_masks = [~d for d in dormant_indices]
    avg_dormancy = np.mean(dormancy_pcts)

    # Baseline eval: nA only
    baseline_masked = agent.evaluate_masked(env_a, head_idx=0)

    # Freeze + recycle + Phase 2
    agent.freeze_active_neurons_and_reset_dormant(0.025)
    agent.set_head(1)
    agent.memory.buffer.clear()
    p2_steps, p2_eps = train_to_convergence(agent, env_b, 150000, get_threshold(env_b), "Phase 2")

    # Final eval
    final_all    = evaluate_standard(agent, env_a, head_idx=0)
    final_masked = agent.evaluate_masked(env_a, head_idx=0)

    print(f"  [W={width}|S{seed}] Dormancy={avg_dormancy:.1f}% | Base(all={baseline_all:.0f} masked={baseline_masked:.0f}) | Final(all={final_all:.0f} masked={final_masked:.0f})")

    return {
        "seed": seed, "width": width,
        "p1_steps": p1_steps, "p1_eps": p1_eps,
        "p2_steps": p2_steps, "p2_eps": p2_eps,
        "dormancy_pct": avg_dormancy,
        "baseline_all": baseline_all, "baseline_masked": baseline_masked,
        "final_all": final_all, "final_masked": final_masked,
    }


def run_experiment_13(widths, num_seeds=5, num_cores=6):
    env_a = "CartPole-v1"
    env_b = "Acrobot-v1"

    print(f"\n{'='*60}")
    print(f"Experiment 13: Width-Varied Activation Masking")
    print(f"Task A: {env_a} → Task B: {env_b}")
    print(f"Widths: {widths} | Seeds: {num_seeds} | Cores: {num_cores}")
    print(f"{'='*60}")

    tasks = [(seed, w, env_a, env_b) for w in widths for seed in range(num_seeds)]

    with mp.Pool(num_cores) as pool:
        raw = pool.starmap(run_single_seed, tasks)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    rows = []
    for w in widths:
        w_results = [r for r in raw if r["width"] == w]

        dormancy    = np.mean([r["dormancy_pct"] for r in w_results])
        p1_steps    = np.mean([r["p1_steps"] for r in w_results])
        p1_eps      = np.mean([r["p1_eps"] for r in w_results])
        p2_steps    = np.mean([r["p2_steps"] for r in w_results])
        p2_eps      = np.mean([r["p2_eps"] for r in w_results])
        b_all       = np.mean([r["baseline_all"] for r in w_results])
        b_masked    = np.mean([r["baseline_masked"] for r in w_results])
        f_all_m     = np.mean([r["final_all"] for r in w_results])
        f_all_s     = np.std([r["final_all"] for r in w_results])
        f_masked_m  = np.mean([r["final_masked"] for r in w_results])
        f_masked_s  = np.std([r["final_masked"] for r in w_results])

        rows.append((w, dormancy, p1_steps, p1_eps, p2_steps, p2_eps,
                     b_all, b_masked, f_all_m, f_all_s, f_masked_m, f_masked_s))

        print(f"Width={w}: nB%={dormancy:.1f}% | P1={p1_steps:.0f}st/{p1_eps:.0f}ep | P2={p2_steps:.0f}st/{p2_eps:.0f}ep")
        print(f"  Baseline: all={b_all:.1f} masked={b_masked:.1f}")
        print(f"  Final:    all={f_all_m:.1f}±{f_all_s:.1f}  masked={f_masked_m:.1f}±{f_masked_s:.1f}")

    with open("exp13_results.md", "w") as f:
        f.write("## Experiment 13: Width-Varied Activation Masking (nA-Only Evaluation)\n")
        f.write(f"**Date:** 2026-08-27 | **Task A:** {env_a} → **Task B:** {env_b} | **Seeds:** {num_seeds}\n\n")
        f.write("| Width | Dormancy (nB%) | P1 Conv (steps/ep) | P2 Conv (steps/ep) | Baseline All | Baseline Masked | Final All | Final Masked |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            w, dorm, p1s, p1e, p2s, p2e, ba, bm, fam, fas, fmm, fms = r
            f.write(f"| {w} | `{dorm:.1f}%` | `{p1s:.0f} / {p1e:.0f}` | `{p2s:.0f} / {p2e:.0f}` | `{ba:.1f}` | `{bm:.1f}` | `{fam:.1f} ± {fas:.1f}` | `{fmm:.1f} ± {fms:.1f}` |\n")

    print(f"\nResults written to exp13_results.md")


def main():
    mp.set_start_method('spawn', force=True)
    run_experiment_13(widths=[64, 128, 256, 512], num_seeds=5, num_cores=6)


if __name__ == "__main__":
    main()
