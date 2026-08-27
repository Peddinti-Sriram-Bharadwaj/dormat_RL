import gymnasium as gym
import torch
import numpy as np
import argparse
from collections import deque

from agents.dqn_agent import DQNAgent
from core.env_wrapper import PadEnvWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Network Width Ablation Study with Multi-Seed")
    parser.add_argument("--env_a", type=str, default="CartPole-v1", help="Phase 1 env")
    parser.add_argument("--env_b", type=str, default="Acrobot-v1", help="Phase 2 env")
    parser.add_argument("--max_steps", type=int, default=150000, help="Max steps per phase")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of random seeds")
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

def train_phase(agent, env_id, max_steps, phase_name, threshold, eval_env_id=None, eval_freq=10, silent=False):
    env = gym.make(env_id)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
    state, _ = env.reset()
    
    epsilon_start = 1.0 if phase_name == "Phase 1" else 0.5
    epsilon_end = 0.05
    epsilon_decay = max_steps // 5
    
    episode_reward = 0
    episode_count = 0
    recent_rewards = deque(maxlen=10)
    
    final_eval = None
    
    if not silent:
        print(f"\n--- Starting {phase_name} on {env_id} ---")
    
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
            
            if not silent and episode_count % 20 == 0:
                print(f"{phase_name} | Step {step} | Episode {episode_count} | Reward {episode_reward} | Mean10 {np.mean(recent_rewards):.1f}")
                
            if eval_env_id is not None and episode_count % eval_freq == 0:
                final_eval = evaluate_agent(agent, eval_env_id)
                if not silent:
                    print(f"    [Eval] Performance on {eval_env_id}: {final_eval:.1f}")
                
            episode_reward = 0
            
            if len(recent_rewards) >= 10 and np.mean(recent_rewards) >= threshold:
                if not silent:
                    print(f"[{phase_name}] Converged at step {step} with mean reward {np.mean(recent_rewards):.1f} >= {threshold}")
                break
            
    env.close()
    
    # Do one last eval if needed
    if eval_env_id is not None:
        final_eval = evaluate_agent(agent, eval_env_id)
        
    return final_eval

def get_threshold(env_id):
    if "CartPole" in env_id:
        return 400.0
    elif "Acrobot" in env_id:
        return -100.0
    elif "MountainCar" in env_id:
        return -110.0
    return 100.0

def main():
    args = parse_args()
    
    widths = [64, 128, 256, 512]
    
    thresh_a = get_threshold(args.env_a)
    thresh_b = get_threshold(args.env_b)
    
    # Store results for logging
    final_stats = {}
    
    for width in widths:
        print(f"\n=========================================")
        print(f"Testing Network Width: {width} (N={args.num_seeds} seeds)")
        print(f"=========================================")
        
        dormancy_list = []
        base_list = []
        final_list = []
        recov_list = []
        
        for seed in range(args.num_seeds):
            print(f"\n--- Seed {seed+1}/{args.num_seeds} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            
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
            train_phase(agent, args.env_a, args.max_steps, "Phase 1", threshold=thresh_a, silent=True)
            phase1_end_perf = evaluate_agent(agent, args.env_a, num_episodes=10)
            
            percentages = agent.freeze_active_neurons_and_reset_dormant(args.dormancy_tau)
            l1_dorm = percentages[1] if len(percentages) > 1 else percentages[0]
                
            # Phase 2
            agent.memory.buffer.clear()
            final_interfered_perf = train_phase(agent, args.env_b, args.max_steps, "Phase 2", threshold=thresh_b, eval_env_id=args.env_a, eval_freq=20, silent=True)
            phase2_end_perf = evaluate_agent(agent, args.env_b, num_episodes=10)
            
            print(f"Seed {seed+1} -> L1 Dorm: {l1_dorm:.1f}%, Base: {phase1_end_perf:.1f}, Final: {phase2_end_perf:.1f}, Recovery: {final_interfered_perf:.1f}")
            
            dormancy_list.append(l1_dorm)
            base_list.append(phase1_end_perf)
            final_list.append(phase2_end_perf)
            recov_list.append(final_interfered_perf)
            
        final_stats[width] = {
            'dormancy_l1': (np.mean(dormancy_list), np.std(dormancy_list)),
            'phase1_taskA': (np.mean(base_list), np.std(base_list)),
            'phase2_taskB': (np.mean(final_list), np.std(final_list)),
            'phase2_taskA_recovery': (np.mean(recov_list), np.std(recov_list))
        }

    print("\n\n=========================================")
    print("ABLATION STUDY RESULTS (Mean ± Std):")
    print("Width | L1 Dormancy | Task A Base | Task B Final | Task A Recovered")
    for w in widths:
        res = final_stats[w]
        print(f"{w:5d} | {res['dormancy_l1'][0]:5.1f}±{res['dormancy_l1'][1]:4.1f}% | {res['phase1_taskA'][0]:6.1f}±{res['phase1_taskA'][1]:4.1f} | {res['phase2_taskB'][0]:6.1f}±{res['phase2_taskB'][1]:4.1f} | {res['phase2_taskA_recovery'][0]:7.1f}±{res['phase2_taskA_recovery'][1]:5.1f}")
    print("=========================================")

if __name__ == "__main__":
    main()
