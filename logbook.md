# Experimental Logbook: Dormant Neurons Phenomenon

## Experiment 1: Baseline CartPole-v1 with High Replay Ratio
**Date:** 2026-08-26
**Environment:** CartPole-v1
**Algorithm:** DQN
**Hyperparameters:** Total Steps=50000, Replay Ratio=2.0, ReDo=True, $\tau$=0.025, ReDo Frequency=1000 steps

**Observations:**
- Training performance was highly unstable and remained poor (mean episodic return $\approx 10-30$).
- The combination of a high replay ratio and frequent ReDo interventions (every 1000 steps) on a low-complexity environment disrupted the network's ability to learn a stable policy.

## Experiment 2: Stabilized CartPole-v1
**Date:** 2026-08-26
**Environment:** CartPole-v1
**Algorithm:** DQN
**Hyperparameters:** Total Steps=50000, Replay Ratio=0.25, ReDo=True, $\tau$=0.025, ReDo Frequency=5000 steps

**Observations:**
- The adjustment of the replay ratio to the standard value (0.25) and the reduction of the ReDo intervention frequency yielded stable policy learning.
- The agent achieved episodic returns upwards of 200 towards the end of training.
- Dormancy was consistently observed despite the simpler environment, with approximately 55% of the neurons in the second hidden layer falling below the dormancy threshold ($\tau \le 0.025$) prior to each ReDo intervention.
- The ReDo algorithm successfully identified and recycled these dormant neurons without inducing catastrophic forgetting under this reduced frequency regime.

## Experiment 3: LunarLander-v3
**Date:** 2026-08-26
**Environment:** LunarLander-v3
**Algorithm:** DQN
**Hyperparameters:** Total Steps=200000, Replay Ratio=0.25, ReDo=True, $\tau$=0.025, ReDo Frequency=5000 steps

**Observations:**
- The environment's increased complexity required a longer training horizon. Initial performance was poor, with episodic returns approximating -200, but exhibited steady improvement over time.
- The agent achieved positive episodic returns starting around step 124,000 and culminated with stable performance exceeding a return of 200 around step 198,000.
- Dormancy accumulation was primarily concentrated in the second hidden layer (Layer 1), starting at roughly 39% early in training and plateauing in the 56% to 60% range during the latter half of the experiment.
- The first hidden layer (Layer 0) maintained a low dormancy rate, generally fluctuating between 1% and 12%.
- Periodic ReDo intervention (every 5000 steps) successfully recycled the significant proportion of dormant neurons in Layer 1, allowing the model to progressively attain a near-solved state for the LunarLander environment without destabilizing the learned policy.

## Experiment 4: Cross-Task Interference and Dormant Neuron Repurposing
**Date:** 2026-08-26
**Environment:** CartPole-v1 (Phase 1) $\rightarrow$ CartPole-v0 (Phase 2)
**Algorithm:** DQN
**Hyperparameters:** Phase 1 Steps=15000, Phase 2 Steps=20000. Phase 1 active neurons frozen; dormant neurons recycled and trained on Phase 2.

**Observations:**
- At the conclusion of Phase 1, the model demonstrated strong performance on CartPole-v1 (episodic returns > 200).
- Following the intervention, active neurons (representing knowledge of CartPole-v1) were frozen, and dormant neurons were reinitialized.
- **Initial Interference:** Upon commencing Phase 2, evaluation on the original Task A (CartPole-v1) exhibited a sharp performance drop (returns collapsed to $\approx 50-60$). This drop did not result from catastrophic forgetting in the active neurons (as they were strictly frozen, verified by $\nabla = 0.0$), but rather from structural interference: the newly randomized dormant neurons were contributing uncalibrated noise to the shared final output layer.
- **Recovery via Repurposing:** As the recycled dormant neurons trained on Task B (CartPole-v0), their outputs aligned with the underlying dynamics of the physics engine. Consequently, not only did the agent successfully solve Task B solely using the newly trained neurons, but evaluation performance on Task A concurrently recovered, climbing back to $\approx 160+$.
- This indicates that dormant neurons can be effectively repurposed for new, similar tasks, and as they adapt, their representations can harmonize with the frozen legacy subnetworks, naturally mitigating the initial forward-interference.

## Experiment 5: Radically Different Cross-Task Transfer (CartPole to Acrobot)
**Date:** 2026-08-26
**Environment:** CartPole-v1 (Phase 1) $\rightarrow$ Acrobot-v1 (Phase 2)
**Algorithm:** DQN
**Hyperparameters:** Phase 1 Steps=15000, Phase 2 Steps=30000. State space zero-padded to dim=8, action space padded to dim=4 to allow architectural continuity. Phase 1 active neurons frozen.

**Observations:**
- **Zero-Padding Mechanism:** The `PadEnvWrapper` successfully allowed a single neural network architecture to interact with two structurally incompatible environments by standardizing their dimensions.
- **Phase 1 Convergence:** The network reached near-perfect episodic returns ($>250$) on `CartPole-v1` before intervention, accumulating significant dormancy ($\approx 58\%$ in Layer 1). 
- **Phase 2 Learning:** The recycled dormant neurons were able to successfully learn features for `Acrobot-v1` despite the active network being entirely locked, improving returns from a baseline of $\approx -459$ up to $-123$.
- **Destructive Forward Interference:** Unlike the structurally similar CartPole-v0 experiment, the evaluation of the frozen CartPole-v1 network collapsed entirely (episodic returns dropping as low as $\approx 9.2$) and *did not recover*. 
- **Conclusion:** While dormant neurons can successfully learn a radically different task, their representations become increasingly misaligned with the original task's frozen network. Because both subnetworks share an output layer, the dormant neurons' specialized adaptations for Acrobot act as highly destructive, irrecoverable noise (forward interference) on the legacy CartPole predictions.
