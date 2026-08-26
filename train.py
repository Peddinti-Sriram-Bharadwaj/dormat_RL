import gymnasium as gym
import torch
import numpy as np
import argparse
import time

from agents.dqn_agent import DQNAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Dormant Neurons Walking Skeleton")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment id")
    parser.add_argument("--total_steps", type=int, default=50000, help="Total environment steps")
    parser.add_argument("--replay_ratio", type=float, default=0.25, help="Gradient updates per env step")
    parser.add_argument("--use_redo", action="store_true", help="Whether to apply ReDo algorithm")
    parser.add_argument("--dormancy_tau", type=float, default=0.025, help="Threshold for dormancy")
    parser.add_argument("--dormancy_eval_freq", type=int, default=1000, help="Steps between dormancy evals")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256], help="Hidden layer dimensions")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, cuda, mps)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize environment
    env = gym.make(args.env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        replay_ratio=args.replay_ratio,
        device=args.device
    )
    
    # Epsilon greedy schedule
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 10000
    
    state, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    
    print(f"Starting training on {args.env}...")
    print(f"Replay Ratio: {args.replay_ratio} | ReDo: {args.use_redo} | Tau: {args.dormancy_tau}")
    
    dormancy_history = []
    
    for step in range(1, args.total_steps + 1):
        # Epsilon calculation
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
            np.exp(-1. * step / epsilon_decay)
            
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
                print(f"Step {step} | Episode {episode_count} | Reward {episode_reward}")
            episode_reward = 0
            
        # Periodic dormancy evaluation and ReDo intervention
        if step % args.dormancy_eval_freq == 0:
            percentages = agent.evaluate_dormancy(
                dormancy_tau=args.dormancy_tau, 
                redo=args.use_redo
            )
            dormancy_history.append((step, percentages))
            
            log_str = f"Step {step} | Dormant Neurons: "
            for i, p in enumerate(percentages):
                log_str += f"Layer {i}: {p:.2f}%  "
            if args.use_redo:
                log_str += "(ReDo applied)"
            print(log_str)
            
    print("Training finished!")
    env.close()

if __name__ == "__main__":
    main()
