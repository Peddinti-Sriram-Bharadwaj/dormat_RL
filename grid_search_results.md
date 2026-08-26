# Grid Search: Cross-Task Transfer & Width Ablation

| Task A | Task B | Width | L1 Dormancy | Task A Base | Task B Final | Task A Recovered |
|---|---|---|---|---|---|---|
| CartPole-v1 | Acrobot-v1 | 64 | 48.44% | 265.0 | -114.0 | 223.6 |
| CartPole-v1 | Acrobot-v1 | 128 | 50.78% | 228.6 | -500.0 | 156.0 |
| CartPole-v1 | Acrobot-v1 | 256 | 57.81% | 246.6 | -269.8 | 26.6 |
| CartPole-v1 | Acrobot-v1 | 512 | 64.06% | 212.4 | -500.0 | 32.8 |
| CartPole-v1 | MountainCar-v0 | 64 | 31.25% | 237.2 | -200.0 | 202.2 |
| CartPole-v1 | MountainCar-v0 | 128 | 43.75% | 247.4 | -200.0 | 10.8 |
| CartPole-v1 | MountainCar-v0 | 256 | 56.25% | 275.6 | -200.0 | 31.2 |
| CartPole-v1 | MountainCar-v0 | 512 | 59.57% | 157.0 | -200.0 | 24.2 |
| Acrobot-v1 | CartPole-v1 | 64 | 39.06% | -500.0 | 36.6 | -500.0 |
| Acrobot-v1 | CartPole-v1 | 128 | 48.44% | -500.0 | 286.6 | -500.0 |
| Acrobot-v1 | CartPole-v1 | 256 | 56.64% | -500.0 | 333.8 | -500.0 |
| Acrobot-v1 | CartPole-v1 | 512 | 73.63% | -500.0 | 135.0 | -500.0 |
| Acrobot-v1 | MountainCar-v0 | 64 | 32.81% | -500.0 | -200.0 | -500.0 |
| Acrobot-v1 | MountainCar-v0 | 128 | 51.56% | -351.0 | -200.0 | -500.0 |
| Acrobot-v1 | MountainCar-v0 | 256 | 62.89% | -500.0 | -200.0 | -500.0 |
| Acrobot-v1 | MountainCar-v0 | 512 | 73.63% | -500.0 | -200.0 | -500.0 |
| MountainCar-v0 | CartPole-v1 | 64 | 65.62% | -200.0 | 318.8 | -200.0 |
| MountainCar-v0 | CartPole-v1 | 128 | 64.84% | -200.0 | 212.8 | -200.0 |
| MountainCar-v0 | CartPole-v1 | 256 | 73.44% | -200.0 | 133.4 | -200.0 |
| MountainCar-v0 | CartPole-v1 | 512 | 76.56% | -200.0 | 127.8 | -200.0 |
| MountainCar-v0 | Acrobot-v1 | 64 | 65.62% | -200.0 | -500.0 | -200.0 |
| MountainCar-v0 | Acrobot-v1 | 128 | 70.31% | -200.0 | -302.0 | -200.0 |
| MountainCar-v0 | Acrobot-v1 | 256 | 73.44% | -200.0 | -173.2 | -200.0 |
| MountainCar-v0 | Acrobot-v1 | 512 | 75.78% | -200.0 | -119.6 | -200.0 |
