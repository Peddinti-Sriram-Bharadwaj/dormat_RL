## Experiment 12: Width-Varied Speed Comparison Grid
**Date:** 2026-08-27 | **Seeds:** 5 | **MountainCar:** velocity reward shaping

| Width | Task A | Task B | Recycled Steps (Mean±Std) | Scratch Steps (Mean±Std) | Δ Steps |
|---|---|---|---|---|---|
| 64 | CartPole-v1 | Acrobot-v1 | `72924 ± 9394`  | `64346 ± 4810` | **-8578** |
| 64 | CartPole-v1 | MountainCar-v0 | `40640 ± 55102`  | `19280 ± 7246` | **-21360** |
| 64 | Acrobot-v1 | CartPole-v1 | `150000 ± 0` (DNF) | `136363 ± 11202` | **-13637** |
| 64 | Acrobot-v1 | MountainCar-v0 | `68120 ± 44901`  | `18480 ± 2685` | **-49640** |
| 64 | MountainCar-v0 | CartPole-v1 | `83844 ± 57606`  | `136360 ± 15035` | **+52516** |
| 64 | MountainCar-v0 | Acrobot-v1 | `66650 ± 15120`  | `61214 ± 5782` | **-5436** |
| 128 | CartPole-v1 | Acrobot-v1 | `55053 ± 5891`  | `56436 ± 3186` | **+1383** |
| 128 | CartPole-v1 | MountainCar-v0 | `13680 ± 5877`  | `13720 ± 3166` | **+40** |
| 128 | Acrobot-v1 | CartPole-v1 | `142761 ± 14478`  | `107953 ± 5771` | **-34808** |
| 128 | Acrobot-v1 | MountainCar-v0 | `17560 ± 10000`  | `14200 ± 1829` | **-3360** |
| 128 | MountainCar-v0 | CartPole-v1 | `146181 ± 4769`  | `112149 ± 9481` | **-34032** |
| 128 | MountainCar-v0 | Acrobot-v1 | `48333 ± 7412`  | `58548 ± 5032` | **+10215** |
| 256 | CartPole-v1 | Acrobot-v1 | `57568 ± 9066`  | `57720 ± 3322` | **+153** |
| 256 | CartPole-v1 | MountainCar-v0 | `7640 ± 3660`  | `9440 ± 1353` | **+1800** |
| 256 | Acrobot-v1 | CartPole-v1 | `107304 ± 48009`  | `94083 ± 21537` | **-13221** |
| 256 | Acrobot-v1 | MountainCar-v0 | `9680 ± 3798`  | `9880 ± 2293` | **+200** |
| 256 | MountainCar-v0 | CartPole-v1 | `111278 ± 23104`  | `91429 ± 7226` | **-19849** |
| 256 | MountainCar-v0 | Acrobot-v1 | `54013 ± 6989`  | `58158 ± 4226` | **+4145** |
| 512 | CartPole-v1 | Acrobot-v1 | `53953 ± 1801`  | `57542 ± 893` | **+3590** |
| 512 | CartPole-v1 | MountainCar-v0 | `4840 ± 1541`  | `7360 ± 833` | **+2520** |
| 512 | Acrobot-v1 | CartPole-v1 | `146916 ± 6168`  | `119564 ± 26471` | **-27352** |
| 512 | Acrobot-v1 | MountainCar-v0 | `6160 ± 1261`  | `6880 ± 776` | **+720** |
| 512 | MountainCar-v0 | CartPole-v1 | `109184 ± 11477`  | `124162 ± 23040` | **+14978** |
| 512 | MountainCar-v0 | Acrobot-v1 | `50247 ± 2307`  | `59626 ± 4334` | **+9379** |
