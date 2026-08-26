import gymnasium as gym
import torch
import numpy as np
from itertools import permutations
import os

from agents.dqn_agent import DQNAgent
from core.env_wrapper import PadEnvWrapper

def evaluate_agent(agent, env_id, num_episodes=5):
    env = gym.make(env_id)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
    total_reward = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
    env.close()
    return total_reward / num_episodes

def train_phase(agent, env_id, total_steps, phase_name, eval_env_id=None, eval_freq=10):
    env = gym.make(env_id)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
    state, _ = env.reset()
    
    epsilon_start = 1.0 if phase_name == "Phase 1" else 0.5
    epsilon_end = 0.05
    epsilon_decay = total_steps // 5
    
    episode_reward = 0
    episode_count = 0
    final_eval = None
    
    for step in range(1, total_steps + 1):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * step / epsilon_decay)
        action = agent.select_action(state, epsilon)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.step(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        
        if done:
            state, _ = env.reset()
            episode_count += 1
            if eval_env_id is not None and episode_count % eval_freq == 0:
                final_eval = evaluate_agent(agent, eval_env_id)
            episode_reward = 0
            
    env.close()
    if eval_env_id is not None and final_eval is None:
        final_eval = evaluate_agent(agent, eval_env_id)
    return final_eval

def append_to_markdown(filename, row_str):
    with open(filename, 'a') as f:
        f.write(row_str + "\n")

def main():
    envs = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
    widths = [64, 128, 256, 512]
    phase1_steps = 15000
    phase2_steps = 30000
    
    output_file = 'grid_search_results.md'
    
    # Initialize markdown table
    with open(output_file, 'w') as f:
        f.write("# Grid Search: Cross-Task Transfer & Width Ablation\n\n")
        f.write("| Task A | Task B | Width | L1 Dormancy | Task A Base | Task B Final | Task A Recovered |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
    pairs = list(permutations(envs, 2))
    
    for env_a, env_b in pairs:
        print(f"\nEvaluating Pair: {env_a} -> {env_b}")
        for width in widths:
            print(f"  > Testing Width: {width}")
            
            # Setup agent
            dummy_env = gym.make(env_a)
            dummy_env = PadEnvWrapper(dummy_env, max_state_dim=8, max_action_dim=4)
            agent = DQNAgent(
                state_dim=dummy_env.observation_space.shape[0],
                action_dim=dummy_env.action_space.n,
                hidden_dims=[width, width],
                replay_ratio=0.25,
                device="cpu"
            )
            dummy_env.close()
            
            # Train Phase 1
            train_phase(agent, env_a, phase1_steps, "Phase 1")
            task_a_base = evaluate_agent(agent, env_a, num_episodes=5)
            
            # Intervention
            percentages = agent.freeze_active_neurons_and_reset_dormant(0.025)
            l1_dormancy = percentages[1] if len(percentages) > 1 else percentages[0]
            agent.memory.buffer.clear()
            
            # Train Phase 2
            task_a_recovered = train_phase(agent, env_b, phase2_steps, "Phase 2", eval_env_id=env_a, eval_freq=20)
            task_b_final = evaluate_agent(agent, env_b, num_episodes=5)
            
            # Write row to file
            row = f"| {env_a} | {env_b} | {width} | {l1_dormancy:.2f}% | {task_a_base:.1f} | {task_b_final:.1f} | {task_a_recovered:.1f} |"
            append_to_markdown(output_file, row)
            print(f"    Completed: {row}")

if __name__ == "__main__":
    main()
