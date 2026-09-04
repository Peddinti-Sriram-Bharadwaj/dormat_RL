"""
Mech-Interp Pilot: Circuit Discovery in a CartPole DQN

Goal: find the small set of hidden neurons that CAUSALLY drives the
"push left" vs "push right" decision at a given state, using two classic
mech-interp tools:

  1. Direct Logit Attribution (DLA) -- a linear, weights-only estimate of
     each neuron's contribution to each output logit:
         effect(neuron, action) = activation[neuron] * W_out[action, neuron]
     Fast, but blind to any nonlinear interaction downstream (there is none
     here since output is a single linear layer on top of the hidden layer,
     so DLA is actually EXACT in this architecture -- a nice property of
     single-hidden-layer MLPs: sum_i effect(i, a) + bias[a] == Q(s,a)).

  2. Activation Patching (ablation) -- zero one neuron at a time, rerun the
     forward pass, and measure the change in the Q-value gap
     (Q(push_right) - Q(push_left)). This is the CAUSAL ground truth check
     against DLA's linear estimate.

  3. Circuit = the smallest set of neurons whose joint ablation collapses
     (or flips) the decision. We also cross-reference this circuit against
     the dormancy mask (tau=0.025) from the existing dormancy pipeline --
     do "dormant" neurons ever show up in the causal circuit? They shouldn't,
     which is a nice sanity check tying mech-interp back to the dormancy work.
"""

import gymnasium as gym
import torch
import numpy as np
import random

from core.network import MLP
from agents.dqn_agent import DQNAgent
from core.dormancy import calculate_dormancy_scores


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_agent(env_name="CartPole-v1", width=256, seed=42, max_steps=150_000, target=475.0):
    set_seed(seed)
    env = gym.make(env_name)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dims=[width],  # single hidden layer -> DLA is exact
        device="cpu",
    )

    ep_returns = []
    state, _ = env.reset(seed=seed)
    ep_ret = 0.0
    steps = 0
    epsilon = 1.0

    while steps < max_steps:
        epsilon = max(0.05, epsilon * 0.9995)
        action = agent.select_action(state, epsilon)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        agent.step(state, action, reward, next_state, done)
        state = next_state
        ep_ret += reward
        steps += 1

        if done:
            ep_returns.append(ep_ret)
            state, _ = env.reset()
            ep_ret = 0.0
            if len(ep_returns) >= 10 and np.mean(ep_returns[-10:]) >= target:
                print(f"  Converged at step {steps} (ep {len(ep_returns)}), mean10={np.mean(ep_returns[-10:]):.1f}")
                break

    env.close()
    return agent


def direct_logit_attribution(net: MLP, state: torch.Tensor):
    """
    Returns:
      q_values: (action_dim,)
      activations: (hidden_dim,) post-ReLU
      effects: (hidden_dim, action_dim) -- effect[i, a] = act[i] * W_out[a, i]
      bias: (action_dim,)
    Exact decomposition for a single-hidden-layer MLP:
      Q(s, a) = sum_i effects[i, a] + bias[a]
    """
    with torch.no_grad():
        q_values, activations = net(state.unsqueeze(0), return_activations=True)
        act = activations[0].squeeze(0)          # (hidden_dim,)
        W_out = net.output_layer.weight          # (action_dim, hidden_dim)
        bias = net.output_layer.bias              # (action_dim,)
        effects = act.unsqueeze(1) * W_out.t()    # (hidden_dim, action_dim)
    return q_values.squeeze(0), act, effects, bias


def ablate_and_measure(net: MLP, state: torch.Tensor, neuron_indices, action_a=1, action_b=0):
    """
    Zero out the given neurons (post-ReLU), rerun the rest of the forward
    pass manually, and return the new Q-gap = Q(action_a) - Q(action_b).
    """
    with torch.no_grad():
        x = state.unsqueeze(0)
        out = x
        for layer in net.layers:
            out = layer(out)
            out = torch.relu(out)
        out = out.clone()
        out[:, neuron_indices] = 0.0
        q = net.output_layer(out).squeeze(0)
    return (q[action_a] - q[action_b]).item()


def run_circuit_discovery(agent: DQNAgent, env_name="CartPole-v1", n_states=5, top_k=8, tau=0.025):
    net = agent.network
    env = gym.make(env_name)
    state, _ = env.reset(seed=123)

    # collect a handful of "interesting" states (mid-episode, pole tilted)
    states = []
    for _ in range(200):
        a = agent.select_action(state, epsilon=0.0)
        next_state, _, term, trunc, _ = env.step(a)
        if abs(state[2]) > 0.03:  # pole angle nontrivial -> decision matters
            states.append(state.copy())
        state = next_state
        if term or trunc:
            state, _ = env.reset()
        if len(states) >= n_states:
            break
    env.close()

    # dormancy mask from a batch sample (reuse existing infra)
    dormant_indices = set()
    if len(agent.memory) >= agent.batch_size:
        s, _, _, _, _ = agent.memory.sample(agent.batch_size)
        s_t = torch.FloatTensor(s)
        with torch.no_grad():
            _, acts = net(s_t, return_activations=True)
        d_idx, d_pct = calculate_dormancy_scores(acts, tau)
        dormant_indices = set(d_idx[0].tolist()) if len(d_idx) else set()
        print(f"[Dormancy] {d_pct[0]:.1f}% dormant ({len(dormant_indices)}/{net.hidden_dims[0]} neurons)\n")

    print("=" * 70)
    print(" CIRCUIT DISCOVERY: CartPole push-left vs push-right decision")
    print("=" * 70)

    for i, s in enumerate(states):
        state_t = torch.FloatTensor(s)
        q, act, effects, bias = direct_logit_attribution(net, state_t)
        gap = (effects[:, 1] - effects[:, 0])  # per-neuron effect on (right - left)
        true_gap = (q[1] - q[0]).item()
        recon_gap = gap.sum().item() + (bias[1] - bias[0]).item()

        chosen = "RIGHT" if q[1] > q[0] else "LEFT"
        print(f"\n--- State {i+1}: pos={s[0]:+.2f} vel={s[1]:+.2f} angle={s[2]:+.3f} angvel={s[3]:+.2f} "
              f"| Q(L)={q[0]:.2f} Q(R)={q[1]:.2f} -> chosen={chosen} ---")
        print(f"  DLA sanity check: sum(neuron effects)+bias = {recon_gap:.4f} vs true Q-gap = {true_gap:.4f}"
              f" (should match exactly)")

        top_pos = torch.topk(gap, top_k).indices.tolist()   # push RIGHT
        top_neg = torch.topk(-gap, top_k).indices.tolist()  # push LEFT

        print(f"  Top {top_k} neurons pushing RIGHT: {top_pos}")
        print(f"  Top {top_k} neurons pushing LEFT:  {top_neg}")
        overlap = set(top_pos + top_neg) & dormant_indices
        print(f"  Overlap with dormant-neuron set: {sorted(overlap) if overlap else 'none'}")

        # causal check: ablate the top-k circuit for the winning direction
        circuit = top_pos if chosen == "RIGHT" else top_neg
        new_gap = ablate_and_measure(net, state_t, circuit)
        flipped = (new_gap > 0) != (true_gap > 0)
        print(f"  [Ablation] Zeroing top-{top_k} circuit neurons -> new Q-gap = {new_gap:+.3f}"
              f" (was {true_gap:+.3f}) -> decision {'FLIPPED' if flipped else 'unchanged'}")

        # control: ablate top_k random non-circuit neurons
        rest = [n for n in range(net.hidden_dims[0]) if n not in circuit]
        random.seed(0)
        control = random.sample(rest, top_k)
        control_gap = ablate_and_measure(net, state_t, control)
        print(f"  [Control]  Zeroing {top_k} random other neurons -> new Q-gap = {control_gap:+.3f}"
              f" (should stay close to {true_gap:+.3f})")


if __name__ == "__main__":
    print("[Phase 1] Training CartPole agent (single hidden layer, width=256)...")
    agent = train_agent()
    run_circuit_discovery(agent)
