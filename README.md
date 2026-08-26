# Dormant Neurons in Reinforcement Learning

This repository contains an implementation and extension of the **Dormant Neuron Phenomenon** (as detailed in works such as *Sokar et al., 2023*) applied to Deep Q-Networks (DQN). The project explores how deep reinforcement learning agents accumulate inactive ("dormant") neurons over training, how resetting them (the ReDo algorithm) can stabilize learning, and how these recycled neurons can be repurposed for completely novel tasks without losing the legacy network's knowledge.

## Features

- **Dormancy Tracking:** A custom PyTorch `MLP` that tracks post-ReLU activations to compute layer-wise dormancy scores during training.
- **ReDo Intervention:** Implementation of the ReDo (Recycle Dormant) algorithm to periodically re-initialize dead neurons and restore network capacity.
- **Cross-Task Transfer:** Advanced scripts to freeze highly active neurons (representing knowledge of a learned task) and train the newly recycled dormant neurons on completely different environments.
- **Structural Padding:** A generic `PadEnvWrapper` that zero-pads state and action spaces, allowing a single frozen neural network architecture to transition between distinct physics engines (e.g., `CartPole-v1` $\rightarrow$ `Acrobot-v1`).
- **Ablation & Grid Search:** Automated scripts to conduct massive cross-task grid searches over various network widths to analyze asymmetrical structural interference.

## Repository Structure

- `core/network.py`: The core MLP architecture built to track forward-pass activations.
- `core/dormancy.py`: Utilities for calculating dormancy scores based on a threshold ($\tau$).
- `core/redo.py`: Logic for applying weight re-initializations (Kaiming He) to identified dormant neurons.
- `core/freeze.py`: PyTorch gradient hooks designed to lock active neurons while allowing recycled neurons to learn freely.
- `core/env_wrapper.py`: The `PadEnvWrapper` for dimension normalization across different Gymnasium environments.
- `agents/dqn_agent.py`: A flexible DQN agent incorporating replay buffers, dormancy evaluation, and active-neuron freezing.

## Running Experiments

You can run the different experimental pipelines using the provided scripts:

1. **Standard ReDo Training:** Train an agent on a single environment with periodic ReDo interventions.
   ```bash
   python train_experiment.py
   ```
2. **Cross-Task Freezing:** Train on Task A, freeze the active sub-network, recycle dormant neurons, and train on Task B.
   ```bash
   python train_freeze_experiment.py
   ```
3. **Width Ablation Grid Search:** Run a comprehensive grid search across multiple network widths and environment permutations to study structural interference.
   ```bash
   python run_grid_search.py
   ```

## Experimental Findings

All empirical observations, hypotheses, and detailed results from the various ablations and grid searches are rigorously documented in academic format.

**Please refer to [`logbook.md`](./logbook.md) for the complete experimental logbook.**
