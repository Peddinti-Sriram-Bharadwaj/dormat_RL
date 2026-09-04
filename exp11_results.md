## Experiment 11: Activation Masking Validation (nA-Only Evaluation)
**Date:** 2026-08-27
**Algorithm:** DQN | Width=256 | Seeds=5
**Task A:** CartPole-v1 → **Task B:** Acrobot-v1

**Phase 1 convergence:** 98041 steps / 1250 episodes
**Phase 2 convergence:** 63053 steps / 316 episodes

| Evaluation Mode | Baseline (pre-Phase 2) | Final (post-Phase 2) |
|---|---|---|
| All neurons ($n_A + n_B$) | `439.4` | `20.7 ± 10.7` |
| nA-only masked ($n_B = 0$) | `349.1` | `198.0 ± 113.2` |

**Interpretation:** If `nA-only masked` collapses even at baseline (before Phase 2), this directly confirms that the dormant neurons ($n_B$) were providing a distributed structural bias to $n_A$ even during Task A training. Removing them breaks the network's internal geometric representation of Task A.
