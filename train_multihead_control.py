import gymnasium as gym
import torch
import numpy as np
from collections import deque
import copy

from agents.dqn_agent import DQNAgent
from core.env_wrapper import PadEnvWrapper
from train_width_ablation import train_phase, evaluate_agent, get_threshold

def run_multihead_experiment(width, num_seeds=3):
    print(f"\n=========================================")
    print(f"Testing Network Width: {width} (N={num_seeds} seeds)")
    print(f"=========================================")
    
    shared_recov = []
    sep_recov = []
    
    env_a = "CartPole-v1"
    env_b = "Acrobot-v1"
    thresh_a = get_threshold(env_a)
    thresh_b = get_threshold(env_b)
    
    for seed in range(num_seeds):
        print(f"\n--- Seed {seed+1}/{num_seeds} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        agent = DQNAgent(
            state_dim=8,
            action_dim=4,
            hidden_dims=[width, width],
            replay_ratio=0.25,
            device="cpu",
            num_heads=2
        )
        
        # --- PHASE 1: Train Task A on Head 0 ---
        agent.set_head(0)
        train_phase(agent, env_a, 150000, "Phase 1", threshold=thresh_a, silent=True)
        phase1_end_perf = evaluate_agent(agent, env_a, num_episodes=10)
        
        # Freeze active neurons (applies to ALL heads)
        percentages = agent.freeze_active_neurons_and_reset_dormant(0.025)
        
        # Save a copy of the agent state for the two branches
        agent_state = copy.deepcopy(agent.network.state_dict())
        target_state = copy.deepcopy(agent.target_network.state_dict())
        opt_state = copy.deepcopy(agent.optimizer.state_dict())
        
        # ==========================================
        # BRANCH 1: SHARED HEAD (Use Head 0 for Task B)
        # ==========================================
        print("  > Running Shared Head Branch...")
        agent.memory.buffer.clear()
        agent.set_head(0)
        
        # Train Task B on Head 0
        train_phase(agent, env_b, 150000, "Phase 2 (Shared)", threshold=thresh_b, silent=True)
        shared_task_b = evaluate_agent(agent, env_b, num_episodes=10)
        
        # Evaluate Task A on Head 0
        agent.set_head(0)
        shared_task_a = evaluate_agent(agent, env_a, num_episodes=10)
        
        shared_recov.append(shared_task_a)
        
        # ==========================================
        # BRANCH 2: SEPARATE HEAD (Use Head 1 for Task B)
        # ==========================================
        print("  > Running Separate Head Branch...")
        # Restore agent state
        agent.network.load_state_dict(agent_state)
        agent.target_network.load_state_dict(target_state)
        agent.optimizer.load_state_dict(opt_state)
        agent.memory.buffer.clear()
        
        # **CRITICAL**: Copy Head 0's weights/biases to Head 1 to ensure a perfectly fair starting point.
        # This ensures the frozen active neurons contribute the exact same initial features to Task B.
        with torch.no_grad():
            agent.network.output_layers[1].weight.copy_(agent.network.output_layers[0].weight)
            agent.network.output_layers[1].bias.copy_(agent.network.output_layers[0].bias)
            agent.target_network.output_layers[1].weight.copy_(agent.target_network.output_layers[0].weight)
            agent.target_network.output_layers[1].bias.copy_(agent.target_network.output_layers[0].bias)
        
        # Train Task B on Head 1
        agent.set_head(1)
        train_phase(agent, env_b, 150000, "Phase 2 (Separate)", threshold=thresh_b, silent=True)
        sep_task_b = evaluate_agent(agent, env_b, num_episodes=10)
        
        # Evaluate Task A on Head 0
        agent.set_head(0)
        sep_task_a = evaluate_agent(agent, env_a, num_episodes=10)
        
        sep_recov.append(sep_task_a)
        
        print(f"    [Shared] Task B: {shared_task_b:.1f} | Task A Recovery: {shared_task_a:.1f}")
        print(f"    [Separate] Task B: {sep_task_b:.1f} | Task A Recovery: {sep_task_a:.1f}")
        
    return shared_recov, sep_recov

def main():
    widths = [256, 512]
    
    for w in widths:
        shared, sep = run_multihead_experiment(w, num_seeds=3)
        print(f"\n--- SUMMARY FOR WIDTH {w} ---")
        print(f"Shared Head Recovery:   {np.mean(shared):.1f} ± {np.std(shared):.1f}")
        print(f"Separate Head Recovery: {np.mean(sep):.1f} ± {np.std(sep):.1f}")
        print("-----------------------------\n")

if __name__ == "__main__":
    main()
