import gymnasium as gym
import torch
import numpy as np
import copy
from collections import deque

from agents.multihead_agent import MultiHeadDQNAgent
from core.env_wrapper import PadEnvWrapper
from train_width_ablation import train_phase, evaluate_agent, get_threshold

def run_lunar_scale_experiment():
    print(f"\n=========================================")
    print(f"Scaling to LunarLander-v3 with ZIR")
    print(f"=========================================")
    
    env_a = "LunarLander-v3"
    env_b = "Acrobot-v1"
    thresh_a = 200.0  # LunarLander solved threshold
    thresh_b = -100.0
    width = 512       # Need high capacity for LunarLander
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Pad to max dimensions of the two envs
    # LunarLander-v3: state_dim=8, action_dim=4
    # Acrobot-v1: state_dim=6, action_dim=3
    # So max_state_dim=8, max_action_dim=4
    
    agent = MultiHeadDQNAgent(
        state_dim=8,
        action_dim=4,
        hidden_dims=[width, width],
        replay_ratio=0.25,
        device="cpu",
        num_heads=2
    )
    
    # --- PHASE 1: Train LunarLander on Head 0 ---
    print("\n[Scaling] Training Phase 1: LunarLander-v3")
    agent.set_head(0)
    # LunarLander takes longer to solve, allow up to 250k steps
    train_phase(agent, env_a, 250000, "Phase 1", threshold=thresh_a, silent=False)
    
    states, _, _, _, _ = agent.memory.sample(agent.batch_size)
    states = torch.FloatTensor(states).to(agent.device)
    with torch.no_grad():
        _, activations = agent.network(states, return_activations=True, head_idx=0)
    
    from core.dormancy import calculate_dormancy_scores
    dormant_indices, percentages = calculate_dormancy_scores(activations, 0.025)
    
    print(f"Dormancy after LunarLander: {percentages}")
    
    agent.freeze_active_neurons_and_reset_dormant(0.025)
    
    print("Evaluating Task A (LunarLander) before Phase 2...")
    agent.set_head(0)
    phase1_end_perf = evaluate_agent(agent, env_a, num_episodes=10)
    print(f"  > LunarLander Baseline: {phase1_end_perf:.1f}")
    
    # ==========================================
    # BRANCH: ZERO-INTERFERENCE ROUTING (ZIR)
    # ==========================================
    print("\n[Scaling] Training Phase 2: Acrobot-v1 with ZIR")
    agent.memory.buffer.clear()
    
    with torch.no_grad():
        # ZIR MAGIC: Completely insulate the Active Sub-Network from the Dormant Sub-Network!
        
        # 1. Isolate Layer 1's Active Neurons from Layer 0's Dormant Neurons
        A1 = ~dormant_indices[1]
        D0 = dormant_indices[0]
        # Create 2D boolean mask of shape (dim1, dim0)
        mask_W1 = A1.unsqueeze(1) & D0.unsqueeze(0)
        agent.network.layers[1].weight[mask_W1] = 0.0
        agent.target_network.layers[1].weight[mask_W1] = 0.0
        
        # 2. Isolate Head 0 from Layer 1's Dormant Neurons
        last_layer_dormant = dormant_indices[-1]
        agent.network.output_layers[0].weight[:, last_layer_dormant] = 0.0
        agent.target_network.output_layers[0].weight[:, last_layer_dormant] = 0.0
        
        # Fair init for Head 1
        agent.network.output_layers[1].weight.copy_(agent.network.output_layers[0].weight)
        agent.network.output_layers[1].bias.copy_(agent.network.output_layers[0].bias)
        agent.target_network.output_layers[1].weight.copy_(agent.target_network.output_layers[0].weight)
        agent.target_network.output_layers[1].bias.copy_(agent.target_network.output_layers[0].bias)
        
    agent.set_head(1)
    train_phase(agent, env_b, 150000, "Phase 2 (ZIR)", threshold=thresh_b, silent=False)
    
    zir_task_b = evaluate_agent(agent, env_b, num_episodes=10)
    
    print("Evaluating Task A (LunarLander) post Phase 2...")
    agent.set_head(0)
    zir_task_a = evaluate_agent(agent, env_a, num_episodes=10)
    
    print(f"\n--- FINAL SCALING RESULTS ---")
    print(f"Phase 1 LunarLander Baseline: {phase1_end_perf:.1f}")
    print(f"Phase 2 Acrobot Final:        {zir_task_b:.1f}")
    print(f"Phase 1 LunarLander Recovery: {zir_task_a:.1f}")

if __name__ == "__main__":
    run_lunar_scale_experiment()
