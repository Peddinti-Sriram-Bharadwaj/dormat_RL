import gymnasium as gym
import torch
import numpy as np
from collections import deque
import copy

from agents.multihead_agent import MultiHeadDQNAgent
from core.env_wrapper import PadEnvWrapper
from train_width_ablation import train_phase, evaluate_agent, get_threshold

import multiprocessing as mp

def run_single_seed(seed, width, env_a, env_b, thresh_a, thresh_b):
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
    
    agent.set_head(0)
    train_phase(agent, env_a, 150000, "Phase 1", threshold=thresh_a, silent=True)
    
    states, _, _, _, _ = agent.memory.sample(agent.batch_size)
    states = torch.FloatTensor(states).to(agent.device)
    with torch.no_grad():
        _, activations = agent.network(states, return_activations=True, head_idx=0)
    
    from core.dormancy import calculate_dormancy_scores
    dormant_indices, percentages = calculate_dormancy_scores(activations, 0.025)
    
    agent.freeze_active_neurons_and_reset_dormant(0.025)
    
    agent.set_head(0)
    evaluate_agent(agent, env_a, num_episodes=10)
    
    agent_state = copy.deepcopy(agent.network.state_dict())
    target_state = copy.deepcopy(agent.target_network.state_dict())
    opt_state = copy.deepcopy(agent.optimizer.state_dict())
    
    # Control Branch
    agent.memory.buffer.clear()
    with torch.no_grad():
        agent.network.output_layers[1].weight.copy_(agent.network.output_layers[0].weight)
        agent.network.output_layers[1].bias.copy_(agent.network.output_layers[0].bias)
        agent.target_network.output_layers[1].weight.copy_(agent.target_network.output_layers[0].weight)
        agent.target_network.output_layers[1].bias.copy_(agent.target_network.output_layers[0].bias)
    
    agent.set_head(1)
    train_phase(agent, env_b, 150000, "Phase 2 (Control)", threshold=thresh_b, silent=True)
    control_task_b = evaluate_agent(agent, env_b, num_episodes=10)
    agent.set_head(0)
    control_task_a = evaluate_agent(agent, env_a, num_episodes=10)
    
    # ZIR Branch
    agent.network.load_state_dict(agent_state)
    agent.target_network.load_state_dict(target_state)
    agent.optimizer.load_state_dict(opt_state)
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
        
        # Copy Head 0's original weights/biases to Head 1 for fair start
        agent.network.output_layers[1].weight.copy_(agent_state["output_layers.0.weight"])
        agent.network.output_layers[1].bias.copy_(agent_state["output_layers.0.bias"])
        agent.target_network.output_layers[1].weight.copy_(target_state["output_layers.0.weight"])
        agent.target_network.output_layers[1].bias.copy_(target_state["output_layers.0.bias"])
        
    agent.set_head(1)
    train_phase(agent, env_b, 150000, "Phase 2 (ZIR)", threshold=thresh_b, silent=True)
    zir_task_b = evaluate_agent(agent, env_b, num_episodes=10)
    agent.set_head(0)
    zir_task_a = evaluate_agent(agent, env_a, num_episodes=10)
    
    return control_task_a, zir_task_a

def run_zir_experiment(width, num_seeds=5, num_cores=6):
    print(f"\n=========================================")
    print(f"Testing ZIR (Zero-Interference Routing) on Width {width} using {num_cores} cores")
    print(f"=========================================")
    
    env_a = "CartPole-v1"
    env_b = "Acrobot-v1"
    thresh_a = get_threshold(env_a)
    thresh_b = get_threshold(env_b)
    
    args = [(seed, width, env_a, env_b, thresh_a, thresh_b) for seed in range(num_seeds)]
    
    with mp.Pool(num_cores) as pool:
        results = pool.starmap(run_single_seed, args)
        
    sep_recov = [r[0] for r in results]
    zir_recov = [r[1] for r in results]
    
    return sep_recov, zir_recov

def main():
    width = 256
    # You have 8 cores, using all but 2 = 6 cores
    sep, zir = run_zir_experiment(width, num_seeds=5, num_cores=6)
    
    print(f"\n--- SUMMARY FOR WIDTH {width} (ZIR VALIDATION) ---")
    print(f"Standard Separate Head Recovery: {np.mean(sep):.1f} ± {np.std(sep):.1f}")
    print(f"ZIR Head Recovery:               {np.mean(zir):.1f} ± {np.std(zir):.1f}")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()
