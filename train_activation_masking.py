"""
Experiment 11: Activation Masking Validation (nA-only evaluation)

Hypothesis: If dormant neurons (nB) provide a distributed structural bias
to the active network (nA), then evaluating Task A using ONLY nA neurons
(zeroing out all nB activations) should cause Task A to collapse —
even though no gradient updates or weight changes have been made.

This is a direct test of the "Structural Bias Loss" theory.
"""

import gymnasium as gym
import torch
import numpy as np
from collections import deque
import multiprocessing as mp

from agents.multihead_agent import MultiHeadDQNAgent
from core.env_wrapper import PadEnvWrapper


def get_threshold(env_id):
    if "CartPole" in env_id:
        return 400.0
    elif "Acrobot" in env_id:
        return -100.0
    elif "LunarLander" in env_id:
        return 200.0
    return 100.0


def train_to_convergence(agent, env_id, max_steps, threshold, phase_name="Phase 1"):
    env = gym.make(env_id)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
    state, _ = env.reset()

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = max_steps // 5

    episode_reward = 0
    episode_count = 0
    recent_rewards = deque(maxlen=10)
    convergence_step = max_steps

    for step in range(1, max_steps + 1):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1.0 * step / epsilon_decay)
        action = agent.select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if done:
            recent_rewards.append(episode_reward)
            state, _ = env.reset()
            episode_count += 1
            episode_reward = 0

            if len(recent_rewards) >= 10 and np.mean(recent_rewards) >= threshold:
                convergence_step = step
                print(f"  [{phase_name}] Converged at step {step} (episode {episode_count}), mean10={np.mean(recent_rewards):.1f}")
                break

    env.close()
    return convergence_step, episode_count


def evaluate_standard(agent, env_id, head_idx=0, n_episodes=10):
    """Standard evaluation using all neurons (nA + nB)."""
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

    print(f"\n[Seed {seed}] Starting: {env_a} -> {env_b}")

    agent = MultiHeadDQNAgent(
        state_dim=8,
        action_dim=4,
        hidden_dims=[width, width],
        replay_ratio=0.25,
        device="cpu",
        num_heads=2
    )

    # --- Phase 1: Train Task A to convergence ---
    agent.set_head(0)
    thresh_a = get_threshold(env_a)
    conv_step_a, conv_ep_a = train_to_convergence(agent, env_a, 150000, thresh_a, "Phase 1")

    # Evaluate Task A baseline with ALL neurons (nA + nB together)
    baseline_all = evaluate_standard(agent, env_a, head_idx=0)

    # Compute dormancy to identify nA masks — but do NOT reset or freeze yet
    from core.dormancy import calculate_dormancy_scores
    states_sample, _, _, _, _ = agent.memory.sample(agent.batch_size)
    states_t = torch.FloatTensor(states_sample)
    with torch.no_grad():
        _, activations = agent.network(states_t, return_activations=True, head_idx=0)
    dormant_indices, _ = calculate_dormancy_scores(activations, 0.025)
    agent.active_masks = [~is_dormant for is_dormant in dormant_indices]

    # Evaluate Task A using ONLY nA neurons (nB zeroed at every layer)
    baseline_masked = agent.evaluate_masked(env_a, head_idx=0)

    print(f"  [Seed {seed}] Phase 1 baseline (all neurons):    {baseline_all:.1f}")
    print(f"  [Seed {seed}] Phase 1 baseline (nA only masked): {baseline_masked:.1f}")

    # --- Freeze nA, recycle nB, train Phase 2 ---
    agent.freeze_active_neurons_and_reset_dormant(0.025)

    agent.set_head(1)
    agent.memory.buffer.clear()
    thresh_b = get_threshold(env_b)
    conv_step_b, conv_ep_b = train_to_convergence(agent, env_b, 150000, thresh_b, "Phase 2")

    # --- Final evaluation after Phase 2 ---
    # Standard eval (all neurons, nA + nB now trained on Task B)
    final_all = evaluate_standard(agent, env_a, head_idx=0)
    # Masked eval (ONLY nA neurons, nB completely zeroed out)
    final_masked = agent.evaluate_masked(env_a, head_idx=0)

    print(f"  [Seed {seed}] Post-Phase2 (all neurons):    {final_all:.1f}")
    print(f"  [Seed {seed}] Post-Phase2 (nA only masked): {final_masked:.1f}")
    print(f"  [Seed {seed}] Task B convergence: {conv_step_b} steps / {conv_ep_b} episodes")

    return {
        "seed": seed,
        "phase1_conv_step": conv_step_a,
        "phase1_conv_ep": conv_ep_a,
        "baseline_all": baseline_all,
        "baseline_masked": baseline_masked,
        "phase2_conv_step": conv_step_b,
        "phase2_conv_ep": conv_ep_b,
        "final_all": final_all,
        "final_masked": final_masked,
    }


def run_experiment_11(width=256, num_seeds=5, num_cores=6):
    env_a = "CartPole-v1"
    env_b = "Acrobot-v1"

    print(f"\n{'='*55}")
    print(f"Experiment 11: Activation Masking Validation")
    print(f"Task A: {env_a}  |  Task B: {env_b}")
    print(f"Width: {width}  |  Seeds: {num_seeds}  |  Cores: {num_cores}")
    print(f"{'='*55}")

    args = [(seed, width, env_a, env_b) for seed in range(num_seeds)]

    with mp.Pool(num_cores) as pool:
        results = pool.starmap(run_single_seed, args)

    print(f"\n{'='*55}")
    print(f"SUMMARY (N={num_seeds} seeds)")
    print(f"{'='*55}")

    baseline_all    = np.mean([r["baseline_all"] for r in results])
    baseline_masked = np.mean([r["baseline_masked"] for r in results])
    final_all       = np.mean([r["final_all"] for r in results])
    final_masked    = np.mean([r["final_masked"] for r in results])

    final_all_std       = np.std([r["final_all"] for r in results])
    final_masked_std    = np.std([r["final_masked"] for r in results])

    p1_conv_steps = np.mean([r["phase1_conv_step"] for r in results])
    p1_conv_eps   = np.mean([r["phase1_conv_ep"] for r in results])
    p2_conv_steps = np.mean([r["phase2_conv_step"] for r in results])
    p2_conv_eps   = np.mean([r["phase2_conv_ep"] for r in results])

    print(f"Phase 1 ({env_a}) convergence:   {p1_conv_steps:.0f} steps / {p1_conv_eps:.0f} episodes")
    print(f"Phase 2 ({env_b}) convergence:   {p2_conv_steps:.0f} steps / {p2_conv_eps:.0f} episodes")
    print()
    print(f"Task A baseline (all neurons):    {baseline_all:.1f}")
    print(f"Task A baseline (nA masked only): {baseline_masked:.1f}")
    print()
    print(f"Task A final (all neurons):       {final_all:.1f} ± {final_all_std:.1f}")
    print(f"Task A final (nA masked only):    {final_masked:.1f} ± {final_masked_std:.1f}")

    # Write results to logbook entry
    with open("exp11_results.md", "w") as f:
        f.write("## Experiment 11: Activation Masking Validation (nA-Only Evaluation)\n")
        f.write(f"**Date:** 2026-08-27\n")
        f.write(f"**Algorithm:** DQN | Width={width} | Seeds={num_seeds}\n")
        f.write(f"**Task A:** {env_a} → **Task B:** {env_b}\n\n")
        f.write(f"**Phase 1 convergence:** {p1_conv_steps:.0f} steps / {p1_conv_eps:.0f} episodes\n")
        f.write(f"**Phase 2 convergence:** {p2_conv_steps:.0f} steps / {p2_conv_eps:.0f} episodes\n\n")
        f.write("| Evaluation Mode | Baseline (pre-Phase 2) | Final (post-Phase 2) |\n")
        f.write("|---|---|---|\n")
        f.write(f"| All neurons ($n_A + n_B$) | `{baseline_all:.1f}` | `{final_all:.1f} ± {final_all_std:.1f}` |\n")
        f.write(f"| nA-only masked ($n_B = 0$) | `{baseline_masked:.1f}` | `{final_masked:.1f} ± {final_masked_std:.1f}` |\n\n")
        f.write("**Interpretation:** If `nA-only masked` collapses even at baseline (before Phase 2), this directly confirms that the dormant neurons ($n_B$) were providing a distributed structural bias to $n_A$ even during Task A training. Removing them breaks the network's internal geometric representation of Task A.\n")

    print(f"\nResults written to exp11_results.md")


def main():
    mp.set_start_method('spawn', force=True)
    run_experiment_11(width=256, num_seeds=5, num_cores=6)


if __name__ == "__main__":
    main()
