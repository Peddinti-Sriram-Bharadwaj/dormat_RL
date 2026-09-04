# Dormant Neurons in Reinforcement Learning

This repository contains an implementation and extension of the **Dormant Neuron Phenomenon** (as detailed in works such as *Sokar et al., 2023*) applied to Deep Q-Networks (DQN). The project explores how deep reinforcement learning agents accumulate inactive ("dormant") neurons over training, how resetting them (the ReDo algorithm) can stabilize learning, and how these recycled neurons can be repurposed for completely novel tasks without losing the legacy network's knowledge.

## Features

- **Dormancy Tracking:** A custom PyTorch `MLP` that tracks post-ReLU activations to compute layer-wise dormancy scores during training.
- **ReDo Intervention:** Implementation of the ReDo (Recycle Dormant) algorithm to periodically re-initialize dead neurons and restore network capacity.
- **Cross-Task Transfer:** Advanced scripts to freeze highly active neurons (representing knowledge of a learned task) and train the newly recycled dormant neurons on completely different environments.
- **Zero-Interference Routing (ZIR):** A novel architectural solution that explicitly isolates a frozen active sub-network from the representation drift of recycled neurons by zeroing out their cascading forward weights, mathematically preventing forward interference.
- **Structural Padding:** A generic `PadEnvWrapper` that zero-pads state and action spaces, allowing a single neural network architecture to transition between distinct physics engines (e.g., `CartPole-v1` $\rightarrow$ `Acrobot-v1`).
- **Rigorous Validation & Scaling:** Automated 5-seed multiprocessing ablation studies, multi-head architectural controls, and scale-up experiments to complex simulators like `LunarLander-v3`.

## Repository Structure

- **Core Mechanics:**
  - `core/network.py`: The core MLP architecture built to track forward-pass activations.
  - `core/dormancy.py`: Utilities for calculating dormancy scores based on a threshold ($\tau$).
  - `core/redo.py`: Logic for applying weight re-initializations (Kaiming He) to identified dormant neurons.
  - `core/freeze.py`: PyTorch gradient hooks designed to lock active neurons while allowing recycled neurons to learn freely.
  - `core/env_wrapper.py`: The `PadEnvWrapper` for dimension normalization.
- **Advanced Architecture (Controls & Solutions):**
  - `core/multihead_network.py` & `agents/multihead_agent.py`: Clean subclasses supporting dynamically routed output heads to isolate output-layer interference.
  - `core/multihead_freeze.py`: Gradient hooks adapted for multi-head isolation.
- **Agents:**
  - `agents/dqn_agent.py`: The foundational DQN agent incorporating replay buffers, dormancy evaluation, and active-neuron freezing.

## Running Experiments

You can run the different experimental pipelines using the provided scripts:

1. **Width Ablation (5-Seed Multiprocessing):** Run a highly rigorous statistical ablation of network capacity using dynamic convergence budgets.
   ```bash
   python train_width_ablation.py
   ```
2. **Multi-Head Interference Control (5-Seed):** Isolate representation drift vs shared-head drift using structurally separated output layers.
   ```bash
   python train_multihead_control.py
   ```
3. **Zero-Interference Routing (ZIR):** Validate the ZIR algorithmic solution, which completely insulates the legacy network from catastrophic noise.
   ```bash
   python train_zir_control.py
   ```
4. **LunarLander Scale-Up:** Test ZIR on a computationally dense continuous physics simulator scaling up to `LunarLander-v3`.
   ```bash
   python train_scale_lunar.py
   ```

## Experimental Findings

All empirical observations, hypotheses, and detailed results from the various ablations, architectural controls, and grid searches are rigorously documented in academic format.

**Please refer to [`logbook.md`](./logbook.md) for the complete experimental logbook.**

## Mechanistic Interpretability: SAE Features & Learned Neuron-Level "Remote Control"

A second line of work extends the dormancy analysis into mechanistic interpretability: instead of just tracking *which* neurons go dormant, we identify *what causal role* the live neurons play, and then use that understanding to steer a frozen agent's behavior from the outside.

### SAE Pilot (`train_sae_pilot.py`)
Trains a TopK Sparse Autoencoder (256 → 1024, k=32) on a converged CartPole DQN's hidden-layer activations, and compares two masking strategies:
- **[B] Coordinate-basis masking** — zero out neurons already flagged dormant (Exp 11/13 approach).
- **[C] SAE-feature masking** — encode → zero dead SAE latents → decode → evaluate on the reconstruction.

**Result:** coordinate-basis masking outperformed SAE masking in this pilot (drop +0.7 vs. drop −27.0, i.e. SAE reconstruction noise dominated). Most of the SAE's degradation traced to imperfect reconstruction (cos_sim=0.999 but still lossy) rather than the masking of dead latents itself — the "masking cost" component was negligible. This does **not** support the superposition hypothesis in this setup; better SAE fidelity would be needed before re-testing. Full log: [`sae_pilot.log`](./sae_pilot.log).

### Circuit Discovery (`circuit_analysis.py`)
On a single-hidden-layer CartPole DQN (linear output layer), **Direct Logit Attribution (DLA)** — `effect(neuron, action) = activation[neuron] * W_out[action, neuron]` — is an *exact* decomposition of each Q-value, not an approximation, since there's no further nonlinearity downstream of the hidden layer. We verified this (`sum(neuron effects) + bias == true Q-gap`, exactly, every state) and then causally confirmed the top-attributed neurons via **activation-patching ablation**:

- A small, **stable** ~8-neuron circuit reproducibly drives the push-left/push-right decision across states (same neuron IDs every time).
- Ablating just those 8 neurons doesn't merely erase the (small) natural Q-gap — it **overshoots and flips the decision hard** (swings of 3–4.5 Q-value points against a baseline gap of ~0.1–0.3), indicating the circuit carries much larger opposing signals that are normally cancelled out elsewhere in the network (a superposition/interference signature).
- Random control neurons, ablated the same way, barely move the Q-gap — confirming circuit specificity.
- The causal circuit and the dormancy-flagged neuron set are disjoint, a useful cross-check with the dormancy pipeline.

Full log: [`circuit_analysis.log`](./circuit_analysis.log).

### Neuron-Level "Remote Control" (`remote_control.py`, `trained_remote_control.py`, `record_remote_control.py`)
Using the circuit above, we built a literal remote control for the frozen agent: clamping the ~8 "RIGHT" neurons high and "LEFT" neurons to zero (or vice versa) forces the argmax action **100% of the time** regardless of the true environment state (`remote_control.py`) — this is the same mechanism as activation-steering/representation-engineering work in the LLM interpretability literature (e.g. clamping SAE features to force behavior), applied here to an RL agent's Q-network.

We then went a step further and **trained a second, independent model** — the *controller* — whose action space is `{AUTO, FORCE_LEFT, FORCE_RIGHT}`. Every step it decides whether to let the frozen base agent balance on its own, or override it via the clamp. It is goal-conditioned on a target cart position and never touches the raw environment action space directly — only the clamp interface (`trained_remote_control.py`):

- **First pass** exposed a reward-hacking failure mode: with only a per-step distance penalty, the controller learned to override constantly and crash the pole quickly (shorter episodes accrue less penalty). Fixed by adding a per-step alive bonus and a much larger one-time crash penalty, so survival dominates the incentive.
- **After retuning** (bigger network, best-checkpoint restoration across training via periodic greedy eval, early stopping on a target survival/accuracy composite): **93.3% survival rate and mean final tracking error of 0.34** over 30 held-out goal positions (track half-length 2.4) — up from ~20% survival in the initial pass.
- `record_remote_control.py` renders a side-by-side video: CartPole balancing on the left, two indicator circles on the right that light up red/blue whenever the controller fires a FORCE_LEFT/FORCE_RIGHT command (see [`remote_control_demo.mp4`](./remote_control_demo.mp4)).

**Context:** this pattern — a learned controller injecting activations into a frozen model to steer it — is an active 2025-2026 research direction (e.g. RL-based steering controllers for vision-language-action models, PID-style activation feedback controllers for LLMs). The specific combination used here — an *exact* linear circuit found via DLA in a toy RL agent, with a goal-conditioned controller and live visualization — doesn't correspond to a specific paper we're aware of, but builds directly on established activation-steering and circuit-discovery techniques.

### Next Steps
Porting the same pipeline (base-agent circuit discovery → clamp interface → goal-conditioned controller) to a MuJoCo continuous-control body, to test whether "macro" commands (walk in a direction, stand, raise a limb) can be issued to a fine-motor-control base policy the same way. In progress on a separate branch.
