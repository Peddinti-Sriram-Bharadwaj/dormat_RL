"""
Record a video of the trained remote-control system in action.

Layout: two halves side by side.
  LEFT:  the actual CartPole environment rendering (the pole balancing).
  RIGHT: two circles side by side (LEFT-command circle, RIGHT-command circle).
         Whenever the controller issues FORCE_RIGHT, the right circle lights
         up. Whenever it issues FORCE_LEFT, the left circle lights up.
         On AUTO, both circles stay dim (no override happening).

Reuses the training code from trained_remote_control.py so the recorded
episode uses a freshly trained base agent + circuit + controller.
"""

import gymnasium as gym
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

from trained_remote_control import (
    train_base_agent, find_circuit, train_controller, AUTO, CMD_LEFT, CMD_RIGHT
)
from remote_control import remote_action, LEFT, RIGHT

PANEL_H = 400
LEFT_W = 600
RIGHT_W = 400
FPS = 30

DIM_LEFT = (60, 40, 40)
DIM_RIGHT = (40, 40, 60)
LIT_LEFT = (230, 70, 70)
LIT_RIGHT = (70, 130, 230)
BG = (18, 18, 22)
TEXT_FG = (230, 230, 230)


def draw_right_panel(command, step, target_x, cur_x):
    img = Image.new("RGB", (RIGHT_W, PANEL_H), BG)
    d = ImageDraw.Draw(img)

    r = 90
    cy = PANEL_H // 2 - 20
    left_cx = RIGHT_W // 2 - 110
    right_cx = RIGHT_W // 2 + 110

    left_color = LIT_LEFT if command == CMD_LEFT else DIM_LEFT
    right_color = LIT_RIGHT if command == CMD_RIGHT else DIM_RIGHT

    d.ellipse([left_cx - r, cy - r, left_cx + r, cy + r], fill=left_color, outline=(90, 90, 90), width=3)
    d.ellipse([right_cx - r, cy - r, right_cx + r, cy + r], fill=right_color, outline=(90, 90, 90), width=3)

    d.text((left_cx - 20, cy - 10), "L", fill=TEXT_FG)
    d.text((right_cx - 20, cy - 10), "R", fill=TEXT_FG)

    label = {AUTO: "AUTO (base agent driving)", CMD_LEFT: "REMOTE: FORCE LEFT", CMD_RIGHT: "REMOTE: FORCE RIGHT"}[command]
    d.text((20, 20), "REMOTE CONTROL", fill=TEXT_FG)
    d.text((20, 45), label, fill=(255, 210, 90) if command != AUTO else (140, 140, 140))
    d.text((20, PANEL_H - 60), f"step {step}", fill=(150, 150, 150))
    d.text((20, PANEL_H - 40), f"target_x={target_x:+.2f}  x={cur_x:+.2f}", fill=(150, 150, 150))

    return img


def compose_frame(cartpole_rgb, command, step, target_x, cur_x):
    left_img = Image.fromarray(cartpole_rgb).resize((LEFT_W, PANEL_H))
    right_img = draw_right_panel(command, step, target_x, cur_x)
    canvas = Image.new("RGB", (LEFT_W + RIGHT_W, PANEL_H), BG)
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (LEFT_W, 0))
    return np.array(canvas)


def rollout_and_record(base_net, right_neurons, left_neurons, controller,
                        target_x, seed, max_steps=250):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state, _ = env.reset(seed=seed)
    frames = []
    for t in range(max_steps):
        obs5 = np.concatenate([state, [target_x]]).astype(np.float32)
        cmd_action = controller.select_action(obs5, epsilon=0.0)
        rgb = env.render()
        frames.append(compose_frame(rgb, cmd_action, t, target_x, state[0]))

        cmd = {AUTO: None, CMD_LEFT: LEFT, CMD_RIGHT: RIGHT}[cmd_action]
        real_action, _ = remote_action(base_net, state, cmd, right_neurons, left_neurons)
        state, _, term, trunc, _ = env.step(real_action)
        if term or trunc:
            # hold last frame briefly so the crash/end is visible
            for _ in range(int(FPS * 0.6)):
                frames.append(frames[-1])
            break
    env.close()
    return frames


if __name__ == "__main__":
    print("[1/4] Training base CartPole agent...")
    base_agent = train_base_agent()
    base_net = base_agent.network
    for p in base_net.parameters():
        p.requires_grad_(False)

    print("[2/4] Finding circuit...")
    right_neurons, left_neurons = find_circuit(base_agent)

    print("[3/4] Training remote-control controller...")
    controller = train_controller(base_net, right_neurons, left_neurons)

    print("[4/4] Recording episode...")
    frames = rollout_and_record(base_net, right_neurons, left_neurons,
                                 controller, target_x=1.3, seed=7, max_steps=250)
    out_path = "remote_control_demo.mp4"
    imageio.mimsave(out_path, frames, fps=FPS, quality=8, macro_block_size=1)
    print(f"Saved {out_path} ({len(frames)} frames, {len(frames)/FPS:.1f}s)")
