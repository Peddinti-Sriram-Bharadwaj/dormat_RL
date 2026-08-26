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
