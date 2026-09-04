"""
Mech-Interp "Remote Control": steer a trained CartPole agent's actions by
directly clamping the causal circuit neurons found via activation patching,
instead of ever touching the environment observation.

Idea:
  From circuit_analysis.py we found a small, STABLE set of neurons that
  drive the push-left vs push-right decision:
      RIGHT-neurons: high activation -> pushes Q(right) up
      LEFT-neurons:  high activation -> pushes Q(left) up

  If we forcibly clamp RIGHT-neurons to a large value and LEFT-neurons to
  zero (or vice versa), we can override the network's decision on demand --
  a "remote control" implemented entirely inside the model's activations,
  with the real sensor input (cart position/velocity/pole angle) irrelevant
  to the outcome.

  We verify this two ways:
    1. Command-following accuracy: sample many random states, send a
       RIGHT / LEFT command via clamping, and check the argmax action
       matches the command (vs. the un-clamped baseline policy).
    2. A live "remote-controlled" episode: run CartPole while alternating
       forced LEFT / forced RIGHT commands on a timer, and confirm the
       agent obeys the remote instead of playing on its own.
"""

import gymnasium as gym
import torch
import numpy as np
import random

from agents.dqn_agent import DQNAgent

LEFT, RIGHT = 0, 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_agent(env_name="CartPole-v1", width=256, seed=42, max_steps=150_000, target=475.0):
    set_seed(seed)
    env = gym.make(env_name)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dims=[width],
        device="cpu",
    )
    ep_returns, state = [], env.reset(seed=seed)[0]
    ep_ret, steps, epsilon = 0.0, 0, 1.0
    while steps < max_steps:
        epsilon = max(0.05, epsilon * 0.9995)
        a = agent.select_action(state, epsilon)
        ns, r, term, trunc, _ = env.step(a)
        done = term or trunc
        agent.step(state, a, r, ns, done)
        state, ep_ret, steps = ns, ep_ret + r, steps + 1
        if done:
            ep_returns.append(ep_ret)
            state, ep_ret = env.reset()[0], 0.0
            if len(ep_returns) >= 10 and np.mean(ep_returns[-10:]) >= target:
                print(f"  Converged at step {steps}, mean10={np.mean(ep_returns[-10:]):.1f}")
                break
    env.close()
    return agent


def find_circuit(agent, env_name="CartPole-v1", n_samples=500, top_k=8):
    """Average DLA over many random on-policy states -> stable circuit."""
    net = agent.network
    env = gym.make(env_name)
    state, _ = env.reset(seed=7)
    gaps = torch.zeros(net.hidden_dims[0])
    n = 0
    while n < n_samples:
        with torch.no_grad():
            q, acts = net(torch.FloatTensor(state).unsqueeze(0), return_activations=True)
            act = acts[0].squeeze(0)
            W_out = net.output_layer.weight  # (2, hidden)
            gaps += act * (W_out[RIGHT] - W_out[LEFT])
        a = agent.select_action(state, epsilon=0.1)
        state, _, term, trunc, _ = env.step(a)
        n += 1
        if term or trunc:
            state, _ = env.reset()
    env.close()
    gaps /= n
    right_neurons = torch.topk(gaps, top_k).indices.tolist()
    left_neurons = torch.topk(-gaps, top_k).indices.tolist()
    print(f"[Circuit] RIGHT-neurons: {right_neurons}")
    print(f"[Circuit] LEFT-neurons:  {left_neurons}")
    return right_neurons, left_neurons


def remote_action(net, state, command, right_neurons, left_neurons, clamp_value=8.0):
    """
    Forward pass with the hidden layer manually clamped to force `command`
    (LEFT or RIGHT), regardless of what `state` actually is.
    command=None -> normal, un-clamped policy (for comparison / "auto" mode).
    """
    with torch.no_grad():
        x = torch.FloatTensor(state).unsqueeze(0)
        out = x
        for layer in net.layers:
            out = layer(out)
            out = torch.relu(out)
        if command is not None:
            out = out.clone()
            if command == RIGHT:
                out[:, right_neurons] = clamp_value
                out[:, left_neurons] = 0.0
            elif command == LEFT:
                out[:, left_neurons] = clamp_value
                out[:, right_neurons] = 0.0
        q = net.output_layer(out).squeeze(0)
        return q.argmax().item(), q


def test_command_following(agent, right_neurons, left_neurons, env_name="CartPole-v1", n_trials=200):
    net = agent.network
    env = gym.make(env_name)
    state, _ = env.reset(seed=99)
    correct = {LEFT: 0, RIGHT: 0}
    total = {LEFT: 0, RIGHT: 0}
    baseline_matches_command = 0

    for i in range(n_trials):
        cmd = RIGHT if i % 2 == 0 else LEFT
        baseline_a, _ = remote_action(net, state, None, right_neurons, left_neurons)
        forced_a, _ = remote_action(net, state, cmd, right_neurons, left_neurons)
        total[cmd] += 1
        correct[cmd] += int(forced_a == cmd)
        baseline_matches_command += int(baseline_a == cmd)
        a = agent.select_action(state, epsilon=0.2)
        state, _, term, trunc, _ = env.step(a)
        if term or trunc:
            state, _ = env.reset()
    env.close()

    acc_left = correct[LEFT] / max(total[LEFT], 1)
    acc_right = correct[RIGHT] / max(total[RIGHT], 1)
    print("\n[Command-Following Test]")
    print(f"  Forced-LEFT obeyed:  {correct[LEFT]}/{total[LEFT]}  ({acc_left*100:.1f}%)")
    print(f"  Forced-RIGHT obeyed: {correct[RIGHT]}/{total[RIGHT]} ({acc_right*100:.1f}%)")
    print(f"  Un-clamped baseline happened to match the alternating command by chance: "
          f"{baseline_matches_command}/{n_trials} ({baseline_matches_command/n_trials*100:.1f}%)")


def run_remote_controlled_episode(agent, right_neurons, left_neurons, env_name="CartPole-v1",
                                   seed=1, hold_steps=15, max_steps=300):
    """
    Live demo: alternate forced LEFT / RIGHT commands every `hold_steps`
    ticks, purely via neuron clamping, and log whether the agent's actual
    action matches the remote command at every step.
    """
    net = agent.network
    env = gym.make(env_name)
    state, _ = env.reset(seed=seed)
    print(f"\n[Live Remote-Controlled Episode] alternating command every {hold_steps} steps")
    obeyed, step = 0, 0
    log = []
    for step in range(max_steps):
        cmd = RIGHT if (step // hold_steps) % 2 == 0 else LEFT
        a, q = remote_action(net, state, cmd, right_neurons, left_neurons)
        obeyed += int(a == cmd)
        state, _, term, trunc, _ = env.step(a)
        log.append((step, "R" if cmd == RIGHT else "L", "R" if a == RIGHT else "L"))
        if term or trunc:
            print(f"  Episode ended at step {step+1} (pole fell / out of bounds) -- "
                  f"expected under forced control since it ignores real state.")
            break
    env.close()
    seq = "".join(c for _, c, _ in log)
    obeyed_seq = "".join(a for _, _, a in log)
    print(f"  Commanded sequence: {seq}")
    print(f"  Actual action seq:  {obeyed_seq}")
    print(f"  Obedience rate: {obeyed}/{len(log)} ({obeyed/len(log)*100:.1f}%)")


if __name__ == "__main__":
    print("[Phase 1] Training CartPole agent...")
    agent = train_agent()

    print("\n[Phase 2] Discovering the push-left/push-right circuit...")
    right_neurons, left_neurons = find_circuit(agent)

    print("\n[Phase 3] Testing remote control via neuron clamping...")
    test_command_following(agent, right_neurons, left_neurons)

    run_remote_controlled_episode(agent, right_neurons, left_neurons)
