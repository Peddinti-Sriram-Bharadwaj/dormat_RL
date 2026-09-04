"""
Atari Dormancy Analysis

Trains a DQN on an Atari game and tracks dormancy across CNN layers
(conv filters AND FC neurons) throughout training.

Output: per-checkpoint dormancy percentages for every layer,
printed as a table and saved to a log file.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from core.atari_cnn import AtariCNN, compute_dormancy
from core.atari_wrappers import make_atari_env


# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32) / 255.0,
            np.array(a),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32) / 255.0,
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── Dormancy Snapshot ─────────────────────────────────────────────────────────
def snapshot_dormancy(network: AtariCNN, replay: ReplayBuffer,
                       batch_size: int = 512, tau: float = 0.025,
                       device: str = "cpu") -> dict:
    """Sample a batch from replay and compute per-layer dormancy."""
    if len(replay) < batch_size:
        return {}
    states, _, _, _, _ = replay.sample(batch_size)
    states_t = torch.FloatTensor(states).to(device)
    with torch.no_grad():
        _, activations = network(states_t, return_activations=True)
    return compute_dormancy(activations, tau=tau)


# ── DQN Training ──────────────────────────────────────────────────────────────
def train_atari_dormancy(
    game_id: str = "ALE/Pong-v5",
    total_steps: int = 500000,
    batch_size: int = 32,
    replay_capacity: int = 100000,
    min_replay: int = 10000,
    target_update_freq: int = 10000,
    dormancy_check_freq: int = 25000,
    lr: float = 1e-4,
    gamma: float = 0.99,
    tau_dormancy: float = 0.025,
    device: str = "cpu",
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = make_atari_env(game_id, clip_rewards=True)
    num_actions = env.action_space.n

    network = AtariCNN(num_actions).to(device)
    target  = AtariCNN(num_actions).to(device)
    target.load_state_dict(network.state_dict())
    target.eval()

    optimizer = optim.Adam(network.parameters(), lr=lr)
    replay    = ReplayBuffer(replay_capacity)

    state, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_count  = 0
    recent_rewards = deque(maxlen=20)

    dormancy_log = []  # list of (step, {layer: pct})

    print(f"\n{'='*60}")
    print(f"Atari Dormancy Analysis: {game_id}")
    print(f"Total steps: {total_steps:,} | Device: {device}")
    print(f"{'='*60}")
    print(f"{'Step':>10} | {'Conv1':>7} | {'Conv2':>7} | {'Conv3':>7} | {'FC1':>7} | {'Mean20Rew':>10}")
    print("-" * 62)

    for step in range(1, total_steps + 1):
        # Epsilon-greedy
        epsilon = max(0.05, 1.0 - (step / (total_steps * 0.5)))
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state[np.newaxis] / 255.0).to(device)
                action = network(state_t).argmax(1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if done:
            recent_rewards.append(episode_reward)
            episode_reward = 0
            episode_count  += 1
            state, _ = env.reset()

        # Train
        if len(replay) >= min_replay:
            states, actions, rewards, next_states, dones = replay.sample(batch_size)
            s  = torch.FloatTensor(states).to(device)
            a  = torch.LongTensor(actions).unsqueeze(1).to(device)
            r  = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            ns = torch.FloatTensor(next_states).to(device)
            d  = torch.FloatTensor(dones).unsqueeze(1).to(device)

            q_vals    = network(s).gather(1, a)
            with torch.no_grad():
                next_q    = target(ns).max(1, keepdim=True)[0]
                target_q  = r + gamma * next_q * (1 - d)

            loss = nn.SmoothL1Loss()(q_vals, target_q)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 10.0)
            optimizer.step()

        # Target update
        if step % target_update_freq == 0:
            target.load_state_dict(network.state_dict())

        # Dormancy snapshot
        if step % dormancy_check_freq == 0 and len(replay) >= min_replay:
            stats = snapshot_dormancy(network, replay, tau=tau_dormancy, device=device)
            if stats:
                row = {k: v["pct"] for k, v in stats.items()}
                dormancy_log.append((step, row))
                mean_rew = np.mean(recent_rewards) if recent_rewards else float("nan")
                print(
                    f"{step:>10,} | "
                    f"{row.get('conv1', 0):>6.1f}% | "
                    f"{row.get('conv2', 0):>6.1f}% | "
                    f"{row.get('conv3', 0):>6.1f}% | "
                    f"{row.get('fc1', 0):>6.1f}% | "
                    f"{mean_rew:>10.2f}"
                )

    env.close()

    # Summary table
    print(f"\n{'='*60}")
    print("FINAL DORMANCY SUMMARY")
    print(f"{'='*60}")
    if dormancy_log:
        final_step, final = dormancy_log[-1]
        for layer, pct in final.items():
            print(f"  {layer:<8}: {pct:.1f}% dormant")

    return dormancy_log


if __name__ == "__main__":
    # Auto-detect best available device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon MPS GPU")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU")

    log = train_atari_dormancy(
        game_id="ALE/Pong-v5",
        total_steps=500000,
        device=device,
        seed=0,
    )
