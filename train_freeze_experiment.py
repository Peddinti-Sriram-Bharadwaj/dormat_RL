import gymnasium as gym
import torch
import numpy as np
import argparse
import time

from agents.dqn_agent import DQNAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Freeze Active Neurons Experiment")
    parser.add_argument("--env_a", type=str, default="CartPole-v1", help="Phase 1 env")
    parser.add_argument("--env_b", type=str, default="CartPole-v0", help="Phase 2 env")
    parser.add_argument("--phase1_steps", type=int, default=20000, help="Steps for Phase 1")
    parser.add_argument("--phase2_steps", type=int, default=30000, help="Steps for Phase 2")
    parser.add_argument("--replay_ratio", type=float, default=0.25, help="Replay ratio")
    parser.add_argument("--dormancy_tau", type=float, default=0.025, help="Threshold")
    parser.add_argument("--device", type=str, default="cpu")
    
    return parser.parse_args()

def evaluate_agent(agent, env_id, num_episodes=5):
    env = gym.make(env_id)
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
    state, _ = env.reset()
    
    epsilon_start = 1.0 if phase_name == "Phase 1" else 0.5
    epsilon_end = 0.05
    epsilon_decay = total_steps // 5
    
    episode_reward = 0
    episode_count = 0
    
    print(f"\n--- Starting {phase_name} on {env_id} ---")
    
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
            if episode_count % 10 == 0:
                print(f"{phase_name} | Step {step} | Episode {episode_count} | Reward {episode_reward}")
                
            if eval_env_id is not None and episode_count % eval_freq == 0:
                eval_reward = evaluate_agent(agent, eval_env_id)
                print(f"    [Eval] Performance on {eval_env_id}: {eval_reward:.1f}")
                
            episode_reward = 0
            
    env.close()

def main():
    args = parse_args()
    
    # Init environment just to get dimensions
    env = gym.make(args.env_a)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        replay_ratio=args.replay_ratio,
        device=args.device
    )
    
    # Phase 1
    train_phase(agent, args.env_a, args.phase1_steps, "Phase 1")
    
    print("\n--- Applying Intervention: Freezing Active Neurons ---")
    percentages = agent.freeze_active_neurons_and_reset_dormant(args.dormancy_tau)
    for i, p in enumerate(percentages):
        print(f"Layer {i}: {p:.2f}% dormant (will be recycled and trained, rest frozen)")
        
    # Phase 2
    # We clear the replay buffer so it doesn't train on Phase 1 data
    agent.memory.buffer.clear()
    
    train_phase(agent, args.env_b, args.phase2_steps, "Phase 2", eval_env_id=args.env_a, eval_freq=10)
    
    print("\nExperiment Finished.")

if __name__ == "__main__":
    main()
