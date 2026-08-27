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
