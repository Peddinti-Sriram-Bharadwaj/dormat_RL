"""
Train a SEPARATE model to BE the remote control.

Previously (remote_control.py) we manually decided when to clamp the
base agent's circuit neurons (hand-scripted LEFT/RIGHT commands).

Here, instead, we train a second, independent DQN -- the "controller" --
whose action space is {AUTO, FORCE_LEFT, FORCE_RIGHT}. Every step it
decides whether to let the frozen base agent play normally, or clamp its
push-left/push-right circuit (found via DLA, same as before) to override
it. The controller is goal-conditioned: it's given a target cart position
and rewarded for driving the cart there WITHOUT letting the pole fall
(since forcing a direction blindly crashes the pole, from the last demo).

This is a clean mech-interp + hierarchical-control result: the controller
never touches the environment's true action space directly -- it only
ever gets to say "clamp these 8 neurons" or "don't." It has to learn to
use the override sparingly and time it against the base agent's own
(better) low-level balancing policy.
"""

import gymnasium as gym
import torch
import numpy as np
import random

from agents.dqn_agent import DQNAgent
from remote_control import train_agent as train_base_agent, find_circuit, remote_action, LEFT, RIGHT

AUTO, CMD_LEFT, CMD_RIGHT = 0, 1, 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RemoteControlEnv(gym.Env):
    """
    Wraps CartPole so the controller's action space is {AUTO, FORCE_LEFT,
    FORCE_RIGHT} instead of the raw {push-left, push-right}. Internally,
    every controller action is translated into a clamp command sent to the
    FROZEN base agent, which then produces the real motor action.

    Observation: [x, x_dot, theta, theta_dot, target_x]
    Reward: -|x - target_x| per step (goal-tracking), minus a crash
            penalty if the pole falls / cart leaves the track early.
    """

    def __init__(self, base_net, right_neurons, left_neurons, track_limit=2.4,
                 max_steps=200, crash_penalty=100.0, alive_bonus=1.0):
        super().__init__()
        self.env = gym.make("CartPole-v1")
        self.base_net = base_net
        self.right_neurons = right_neurons
        self.left_neurons = left_neurons
        self.track_limit = track_limit
        self.max_steps = max_steps
        self.crash_penalty = crash_penalty
        self.alive_bonus = alive_bonus
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

    def _obs(self, state):
        return np.concatenate([state, [self.target_x]]).astype(np.float32)

    def reset(self, seed=None, **kwargs):
        state, _ = self.env.reset(seed=seed)
        self.target_x = np.random.uniform(-1.8, 1.8)
        self.t = 0
        self._state = state
        return self._obs(state), {}

    def step(self, controller_action):
        cmd = {AUTO: None, CMD_LEFT: LEFT, CMD_RIGHT: RIGHT}[controller_action]
        real_action, _ = remote_action(self.base_net, self._state, cmd,
                                        self.right_neurons, self.left_neurons)
        next_state, _, term, trunc, _ = self.env.step(real_action)
        self.t += 1

        dist = abs(next_state[0] - self.target_x)
        reward = self.alive_bonus - dist * 0.5
        crashed = term  # CartPole terminates on angle/track-limit violation
        if crashed:
            reward -= self.crash_penalty
        done = term or trunc or self.t >= self.max_steps

        self._state = next_state
        return self._obs(next_state), reward, done, False, {"crashed": crashed, "dist": dist}

    def close(self):
        self.env.close()


def greedy_eval(base_net, right_neurons, left_neurons, controller, n_episodes=20, seed_base=5000):
    """Epsilon=0 rollout to measure true performance, decoupled from exploration noise."""
    env = RemoteControlEnv(base_net, right_neurons, left_neurons)
    survived, dists, autos = 0, [], []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed_base + i)
        crashed, steps, auto_ct = False, 0, 0
        while True:
            a = controller.select_action(state, epsilon=0.0)
            auto_ct += int(a == AUTO)
            state, r, done, _, info = env.step(a)
            steps += 1
            if done:
                crashed = info["crashed"]
                break
        survived += int(not crashed)
        dists.append(info["dist"])
        autos.append(auto_ct / steps)
    env.close()
    return survived / n_episodes, float(np.mean(dists)), float(np.mean(autos))


def train_controller(base_net, right_neurons, left_neurons, max_steps=400_000, seed=0,
                      eval_every=100, patience_evals=6, target_survival=0.9, target_dist=0.35):
    set_seed(seed)
    env = RemoteControlEnv(base_net, right_neurons, left_neurons)
    controller = DQNAgent(
        state_dim=5,
        action_dim=3,
        hidden_dims=[128, 128],
        lr=5e-4,
        buffer_size=100_000,
        batch_size=128,
        replay_ratio=0.5,
        device="cpu",
    )

    state, _ = env.reset(seed=seed)
    ep_ret, ep_returns, steps, epsilon = 0.0, [], 0, 1.0
    best_score, best_state, evals_no_improve = -1.0, None, 0

    while steps < max_steps:
        epsilon = max(0.03, epsilon * 0.99995)
        a = controller.select_action(state, epsilon)
        ns, r, done, _, info = env.step(a)
        controller.step(state, a, r, ns, done)
        state, ep_ret, steps = ns, ep_ret + r, steps + 1
        if done:
            ep_returns.append(ep_ret)
            state, _ = env.reset()
            ep_ret = 0.0

            if len(ep_returns) % eval_every == 0:
                surv, dist, auto_frac = greedy_eval(base_net, right_neurons, left_neurons, controller)
                score = surv - dist * 0.2  # composite: prioritize not crashing, then accuracy
                print(f"  step {steps:>7d} | ep {len(ep_returns):>5d} | train mean50={np.mean(ep_returns[-50:]):7.1f} "
                      f"| eval: survival={surv*100:5.1f}% mean_dist={dist:.3f} auto={auto_frac*100:.0f}% "
                      f"| epsilon={epsilon:.3f}")

                if score > best_score:
                    best_score = score
                    best_state = {k: v.clone() for k, v in controller.network.state_dict().items()}
                    evals_no_improve = 0
                else:
                    evals_no_improve += 1

                if surv >= target_survival and dist <= target_dist:
                    print(f"  -> reached target performance (survival>={target_survival*100:.0f}%, "
                          f"dist<={target_dist}); stopping early.")
                    break
                if evals_no_improve >= patience_evals and steps > max_steps * 0.3:
                    print(f"  -> no improvement for {patience_evals} evals; stopping early.")
                    break

    if best_state is not None:
        controller.network.load_state_dict(best_state)
        controller.target_network.load_state_dict(best_state)
        print(f"  Restored best checkpoint (composite score={best_score:.3f}).")

    env.close()
    return controller


def evaluate(base_net, right_neurons, left_neurons, controller, n_episodes=30):
    env = RemoteControlEnv(base_net, right_neurons, left_neurons)
    cmd_names = {AUTO: "AUTO", CMD_LEFT: "L", CMD_RIGHT: "R"}
    print("\n[Evaluation] goal-conditioned control (target cart position)")
    survived, dists = 0, []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=1000 + ep)
        target = env.target_x
        cmds_used, final_dist, crashed, t = [], None, False, 0
        while True:
            a = controller.select_action(state, epsilon=0.0)
            cmds_used.append(cmd_names[a])
            state, r, done, _, info = env.step(a)
            t += 1
            final_dist = info["dist"]
            if done:
                crashed = info["crashed"]
                break
        auto_pct = cmds_used.count("AUTO") / len(cmds_used) * 100
        survived += int(not crashed)
        dists.append(final_dist)
        print(f"  ep{ep:02d} target_x={target:+.2f} | steps={t:3d} | final|x-target|={final_dist:.2f} "
              f"| crashed={crashed} | AUTO used {auto_pct:.0f}% of steps")
    env.close()
    print(f"\n  SUMMARY over {n_episodes} episodes: survival rate = {survived}/{n_episodes} "
          f"({survived/n_episodes*100:.1f}%), mean final |x-target| = {np.mean(dists):.3f}")


if __name__ == "__main__":
    print("[Phase 1] Training the base CartPole agent (to be remote-controlled)...")
    base_agent = train_base_agent()
    base_net = base_agent.network
    for p in base_net.parameters():
        p.requires_grad_(False)  # freeze -- controller must work only via clamping

    print("\n[Phase 2] Finding the push-left/push-right circuit in the base agent...")
    right_neurons, left_neurons = find_circuit(base_agent)

    print("\n[Phase 3] Training the CONTROLLER (a separate model) to remote-control it...")
    controller = train_controller(base_net, right_neurons, left_neurons)

    evaluate(base_net, right_neurons, left_neurons, controller)
