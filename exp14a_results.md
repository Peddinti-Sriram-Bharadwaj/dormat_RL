## Experiment 14A: Random Neuron Masking Control
**Date:** 2026-08-27 | **Task A:** CartPole-v1 | **Seeds:** 5

**Interpretation:** If `Drop(dormant) ≈ Drop(random)`, the structural bias effect is explained by general capacity loss, not dormancy-specific structure. If `Drop(dormant) < Drop(random)`, dormant neurons carry LESS useful structure than active ones (expected). If `Drop(dormant) > Drop(random)`, dormant neurons carry MORE structure than expected from random neurons of equal count — this would be the strongest evidence for the structural bias claim.

| Width | nB% | All Neurons | Dormant-Masked (Mean±Std) | Random-Masked (Mean±Std) | Drop(dormant) | Drop(random) |
|---|---|---|---|---|---|---|
| 64 | `31.7%` | `395.8` | `391.5 ± 142.8` | `105.6 ± 67.2` | `4.3` | `290.2` |
| 128 | `33.2%` | `372.0` | `310.3 ± 194.3` | `52.3 ± 28.2` | `61.7` | `319.7` |
| 256 | `38.4%` | `416.4` | `425.9 ± 148.2` | `62.1 ± 19.6` | `-9.6` | `354.2` |
| 512 | `47.4%` | `425.4` | `406.8 ± 166.7` | `86.3 ± 43.9` | `18.6` | `339.1` |
