"""
Experiment 14A: Random Neuron Masking Control

Critical baseline for the "structural bias" claim.

Hypothesis to test: Is the performance drop when masking nB neurons
due to something SPECIAL about dormant neurons, or just general capacity loss?

Method:
  For each seed after Phase 1 training:
  1. Measure Task A with ALL neurons (standard baseline)
  2. Measure Task A masking DORMANT neurons (nB masked) — our Exp 11 result
  3. Measure Task A masking a RANDOM set of the same number of neurons — new control
  4. Compare: if (3) drops as much as (2), the effect is just capacity loss, not dormancy-specific.
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
    return 100.0


def train_to_convergence(agent, env_id, max_steps, threshold):
    env = gym.make(env_id)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
    state, _ = env.reset()

    epsilon_start = 1.0
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
                break

    env.close()


def evaluate_with_mask(agent, env_id, masks, head_idx=0, n_episodes=10):
    """Evaluate using a specific activation mask (zeros out masked neurons)."""
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
                q_values = agent.network.forward_masked(state_t, masks, head_idx=head_idx)
            action = q_values.argmax(dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            ep_r += reward
            done = terminated or truncated
        total += ep_r

    env.close()
    return total / n_episodes


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


def run_single_seed(seed, width, env_a, n_random_repeats=5):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = MultiHeadDQNAgent(
        state_dim=8, action_dim=4, hidden_dims=[width, width],
        replay_ratio=0.25, device="cpu", num_heads=1
    )
    agent.set_head(0)
    train_to_convergence(agent, env_a, 150000, get_threshold(env_a))

    # Standard baseline: all neurons
    score_all = evaluate_standard(agent, env_a)

    # Identify dormant neurons
    states_sample, _, _, _, _ = agent.memory.sample(agent.batch_size)
    states_t = torch.FloatTensor(states_sample)
    with torch.no_grad():
        _, activations = agent.network(states_t, return_activations=True, head_idx=0)
    dormant_indices, dormancy_pcts = calculate_dormancy_scores(activations, 0.025)

    # Count dormant neurons per layer
    n_dormant_per_layer = [d.sum().item() for d in dormant_indices]
    avg_dormancy = np.mean(dormancy_pcts)

    # nB-masked: mask exactly the dormant neurons
    active_masks = [~d for d in dormant_indices]
    score_dormant_masked = evaluate_with_mask(agent, env_a, active_masks)

    # Random-masked: mask the SAME NUMBER of neurons, but chosen randomly
    random_masked_scores = []
    for _ in range(n_random_repeats):
        random_masks = []
        for layer_idx, n_mask in enumerate(n_dormant_per_layer):
            layer_width = width
            perm = torch.randperm(layer_width)
            mask = torch.ones(layer_width, dtype=torch.bool)
            mask[perm[:int(n_mask)]] = False  # mask random neurons
            random_masks.append(mask)
        score = evaluate_with_mask(agent, env_a, random_masks)
        random_masked_scores.append(score)

    score_random_masked = np.mean(random_masked_scores)

    print(f"  [W={width}|S{seed}] Dormancy={avg_dormancy:.1f}% nDormant={n_dormant_per_layer}")
    print(f"    All: {score_all:.1f} | Dormant-masked: {score_dormant_masked:.1f} | Random-masked: {score_random_masked:.1f}")

    return {
        "seed": seed, "width": width,
        "dormancy_pct": avg_dormancy,
        "n_dormant": n_dormant_per_layer,
        "score_all": score_all,
        "score_dormant_masked": score_dormant_masked,
        "score_random_masked": score_random_masked,
    }


def run_experiment_14a(widths, num_seeds=5, num_cores=6):
    env_a = "CartPole-v1"

    print(f"\n{'='*60}")
    print(f"Experiment 14A: Random Masking Control")
    print(f"Task A: {env_a} | Widths: {widths} | Seeds: {num_seeds}")
    print(f"{'='*60}")

    tasks = [(seed, w, env_a) for w in widths for seed in range(num_seeds)]

    with mp.Pool(num_cores) as pool:
        raw = pool.starmap(run_single_seed, tasks)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    rows = []
    for w in widths:
        w_results = [r for r in raw if r["width"] == w]

        dorm        = np.mean([r["dormancy_pct"] for r in w_results])
        score_all   = np.mean([r["score_all"] for r in w_results])
        score_dm    = np.mean([r["score_dormant_masked"] for r in w_results])
        score_dm_s  = np.std([r["score_dormant_masked"] for r in w_results])
        score_rm    = np.mean([r["score_random_masked"] for r in w_results])
        score_rm_s  = np.std([r["score_random_masked"] for r in w_results])

        drop_dormant = score_all - score_dm
        drop_random  = score_all - score_rm

        rows.append((w, dorm, score_all, score_dm, score_dm_s, score_rm, score_rm_s, drop_dormant, drop_random))
        print(f"Width={w}: nB%={dorm:.1f}% | All={score_all:.1f} | Dormant-masked={score_dm:.1f}±{score_dm_s:.1f} | Random-masked={score_rm:.1f}±{score_rm_s:.1f}")
        print(f"  Drop(dormant)={drop_dormant:.1f}  Drop(random)={drop_random:.1f}")

    with open("exp14a_results.md", "w") as f:
        f.write("## Experiment 14A: Random Neuron Masking Control\n")
        f.write(f"**Date:** 2026-08-27 | **Task A:** {env_a} | **Seeds:** {num_seeds}\n\n")
        f.write("**Interpretation:** If `Drop(dormant) ≈ Drop(random)`, the structural bias effect is explained by general capacity loss, not dormancy-specific structure. If `Drop(dormant) < Drop(random)`, dormant neurons carry LESS useful structure than active ones (expected). If `Drop(dormant) > Drop(random)`, dormant neurons carry MORE structure than expected from random neurons of equal count — this would be the strongest evidence for the structural bias claim.\n\n")
        f.write("| Width | nB% | All Neurons | Dormant-Masked (Mean±Std) | Random-Masked (Mean±Std) | Drop(dormant) | Drop(random) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            w, dorm, sa, sdm, sdms, srm, srms, dd, dr = r
            f.write(f"| {w} | `{dorm:.1f}%` | `{sa:.1f}` | `{sdm:.1f} ± {sdms:.1f}` | `{srm:.1f} ± {srms:.1f}` | `{dd:.1f}` | `{dr:.1f}` |\n")

    print(f"\nResults written to exp14a_results.md")


def main():
    mp.set_start_method('spawn', force=True)
    run_experiment_14a(widths=[64, 128, 256, 512], num_seeds=5, num_cores=6)


if __name__ == "__main__":
    main()
