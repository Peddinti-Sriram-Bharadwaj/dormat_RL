import gymnasium as gym
import torch
import numpy as np
import copy
from collections import deque
import multiprocessing as mp

from agents.multihead_agent import MultiHeadDQNAgent
from core.env_wrapper import PadEnvWrapper

def get_threshold(env_id):
    if "CartPole" in env_id:
        return 400.0
    elif "Acrobot" in env_id:
        return -100.0
    elif "MountainCar" in env_id:
        return -110.0
    elif "LunarLander" in env_id:
        return 200.0
    return 100.0

def train_phase_for_speed(agent, env_id, max_steps, phase_name, threshold):
    env = gym.make(env_id)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
    state, _ = env.reset()
    
    epsilon_start = 1.0 if phase_name == "Phase 1" or phase_name == "Scratch" else 0.5
    epsilon_end = 0.05
    epsilon_decay = max_steps // 5
    
    episode_reward = 0
    episode_count = 0
    recent_rewards = deque(maxlen=10)
    
    for step in range(1, max_steps + 1):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * step / epsilon_decay)
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
                env.close()
                return step  # Return the exact step it converged!
            
    env.close()
    return max_steps  # Failed to converge

def run_single_seed(seed, width, env_a, env_b, thresh_a, thresh_b):
    torch.set_num_threads(1)
    print(f"\n--- Seed {seed} ---")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    agent = MultiHeadDQNAgent(
        state_dim=8,
        action_dim=4,
        hidden_dims=[width, width],
        replay_ratio=0.25,
        device="cpu",
        num_heads=2
    )
    
    # --- PHASE 1: Train Task A ---
    agent.set_head(0)
    train_phase_for_speed(agent, env_a, 150000, "Phase 1", threshold=thresh_a)
    
    # Freeze and reset dormant
    agent.freeze_active_neurons_and_reset_dormant(0.025)
    
    # --- PHASE 2: Train Task B (Recycled) ---
    print(f"[Seed {seed}] Training Recycled Network on {env_b}...")
    agent.set_head(1)
    agent.memory.buffer.clear()
    steps_recycled = train_phase_for_speed(agent, env_b, 150000, "Recycled", threshold=thresh_b)
    
    # --- SCRATCH: Train Task B from Scratch ---
    print(f"[Seed {seed}] Training Scratch Network on {env_b}...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    scratch_agent = MultiHeadDQNAgent(
        state_dim=8,
        action_dim=4,
        hidden_dims=[width, width],
        replay_ratio=0.25,
        device="cpu",
        num_heads=1
    )
    scratch_agent.set_head(0)
    steps_scratch = train_phase_for_speed(scratch_agent, env_b, 150000, "Scratch", threshold=thresh_b)
    
    print(f"[Seed {seed}] Recycled: {steps_recycled} steps | Scratch: {steps_scratch} steps")
    return steps_recycled, steps_scratch

import itertools

def run_speed_experiment_full_grid(width=256, num_seeds=5, num_cores=6):
    print(f"\n=========================================")
    print(f"Speed Comparison Grid Search (Width {width})")
    print(f"=========================================")
    
    games = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    permutations = list(itertools.permutations(games, 2))
    
    tasks = []
    for env_a, env_b in permutations:
        for seed in range(num_seeds):
            thresh_a = get_threshold(env_a)
            thresh_b = get_threshold(env_b)
            tasks.append((seed, width, env_a, env_b, thresh_a, thresh_b))
            
    print(f"Total parallel tasks queued: {len(tasks)} (running on {num_cores} cores)")
    
    with mp.Pool(num_cores) as pool:
        raw_results = pool.starmap(run_single_seed, tasks)
        
    print("\n--- FINAL GRID RESULTS ---")
    
    with open("speed_grid_results.md", "w") as f:
        f.write("# Speed Comparison Grid Search\n\n")
        f.write("| Task A | Task B | Recycled Convergence (Mean ± Std) | Scratch Convergence (Mean ± Std) | Faster By |\n")
        f.write("|---|---|---|---|---|\n")
        
        task_idx = 0
        for env_a, env_b in permutations:
            recycled = []
            scratch = []
            for seed in range(num_seeds):
                r, s = raw_results[task_idx]
                recycled.append(r)
                scratch.append(s)
                task_idx += 1
                
            r_mean, r_std = np.mean(recycled), np.std(recycled)
            s_mean, s_std = np.mean(scratch), np.std(scratch)
            faster_by = s_mean - r_mean
            
            f.write(f"| {env_a} | {env_b} | {r_mean:.1f} ± {r_std:.1f} | {s_mean:.1f} ± {s_std:.1f} | {faster_by:.1f} |\n")
            print(f"{env_a} -> {env_b} | Recycled: {r_mean:.1f} | Scratch: {s_mean:.1f} | Diff: {faster_by:.1f}")

def main():
    mp.set_start_method('spawn', force=True)
    width = 256
    run_speed_experiment_full_grid(width, num_seeds=5, num_cores=6)

if __name__ == "__main__":
    main()
