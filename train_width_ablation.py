import gymnasium as gym
import torch
import numpy as np
import argparse

from agents.dqn_agent import DQNAgent
from core.env_wrapper import PadEnvWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Network Width Ablation Study")
    parser.add_argument("--env_a", type=str, default="CartPole-v1", help="Phase 1 env")
    parser.add_argument("--env_b", type=str, default="Acrobot-v1", help="Phase 2 env")
    parser.add_argument("--phase1_steps", type=int, default=15000, help="Steps for Phase 1")
    parser.add_argument("--phase2_steps", type=int, default=30000, help="Steps for Phase 2")
    parser.add_argument("--replay_ratio", type=float, default=0.25, help="Replay ratio")
    parser.add_argument("--dormancy_tau", type=float, default=0.025, help="Threshold")
    parser.add_argument("--device", type=str, default="cpu")
    
    return parser.parse_args()

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

def train_phase(agent, env_id, total_steps, phase_name, eval_env_id=None, eval_freq=10, silent=False):
    env = gym.make(env_id)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
    state, _ = env.reset()
    
    epsilon_start = 1.0 if phase_name == "Phase 1" else 0.5
    epsilon_end = 0.05
    epsilon_decay = total_steps // 5
    
    episode_reward = 0
    episode_count = 0
    
    final_eval = None
    
    if not silent:
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
            
            if not silent and episode_count % 20 == 0:
                print(f"{phase_name} | Step {step} | Episode {episode_count} | Reward {episode_reward}")
                
            if eval_env_id is not None and episode_count % eval_freq == 0:
                final_eval = evaluate_agent(agent, eval_env_id)
                if not silent:
                    print(f"    [Eval] Performance on {eval_env_id}: {final_eval:.1f}")
                
            episode_reward = 0
            
    env.close()
    
    # Do one last eval if needed
    if eval_env_id is not None and final_eval is None:
        final_eval = evaluate_agent(agent, eval_env_id)
        
    return final_eval

def main():
    args = parse_args()
    
    widths = [64, 128, 256, 512]
    
    # Store results for logging
    results = {}
    
    for width in widths:
        print(f"\n=========================================")
        print(f"Testing Network Width: {width}")
        print(f"=========================================")
        
        # Init dummy environment to get padded dimensions
        env = gym.make(args.env_a)
        env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()
        
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[width, width],
            replay_ratio=args.replay_ratio,
            device=args.device
        )
        
        # Phase 1
        train_phase(agent, args.env_a, args.phase1_steps, "Phase 1")
        phase1_end_perf = evaluate_agent(agent, args.env_a, num_episodes=10)
        print(f"> Phase 1 End Performance ({args.env_a}): {phase1_end_perf:.1f}")
        
        percentages = agent.freeze_active_neurons_and_reset_dormant(args.dormancy_tau)
        print("\n--- Applying Intervention: Freezing Active Neurons ---")
        for i, p in enumerate(percentages):
            print(f"Layer {i}: {p:.2f}% dormant (will be recycled and trained, rest frozen)")
            
        # Phase 2
        agent.memory.buffer.clear()
        
        # To avoid too much output, make phase 2 silent but capture final evaluations
        final_interfered_perf = train_phase(agent, args.env_b, args.phase2_steps, "Phase 2", eval_env_id=args.env_a, eval_freq=20, silent=False)
        phase2_end_perf = evaluate_agent(agent, args.env_b, num_episodes=10)
        
        print(f"\n> Final Performance on {args.env_b} (Task B): {phase2_end_perf:.1f}")
        print(f"> Final Recovered Performance on {args.env_a} (Task A): {final_interfered_perf:.1f}")
        
        results[width] = {
            'dormancy_l1': percentages[1] if len(percentages) > 1 else percentages[0],
            'phase1_taskA': phase1_end_perf,
            'phase2_taskB': phase2_end_perf,
            'phase2_taskA_recovery': final_interfered_perf
        }

    print("\n\n=========================================")
    print("ABLATION STUDY RESULTS:")
    print("Width | L1 Dormancy | Task A Base | Task B Final | Task A Recovered")
    for w in widths:
        res = results[w]
        print(f"{w:5d} | {res['dormancy_l1']:10.2f}% | {res['phase1_taskA']:11.1f} | {res['phase2_taskB']:12.1f} | {res['phase2_taskA_recovery']:16.1f}")
    print("=========================================")

if __name__ == "__main__":
    main()
