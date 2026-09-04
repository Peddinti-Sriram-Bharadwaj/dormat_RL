"""
SAE Pilot Experiment: Sparse Autoencoder for Feature-Level Activation Isolation

Goal: Compare coordinate-basis dormant neuron masking (Exp 11/13 approach) vs.
feature-space masking using a trained TopK SAE.

Hypothesis: SAE-masked evaluation (zeroing out dead SAE latents then reconstructing)
preserves Task A performance BETTER than coordinate-axis nB masking,
because SAE features are monosemantic and sparse, and the active latents
span the true task-relevant feature directions rather than axis-aligned subspaces.

Pipeline:
  1. Train CartPole DQN agent to convergence (Phase 1)
  2. Dump 10,000 post-ReLU activations from replay buffer (last hidden layer)
  3. Train TopK SAE offline on those activations (d -> m, expansion 4x)
  4. Three evaluations:
       A) Baseline: all neurons active
       B) Coord-basis: zero out dormant neurons (existing Exp 11/13 approach)
       C) SAE-masked: encode -> zero dead latents -> decode -> use reconstructed act
  5. Report comparison table
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

from core.multihead_network import MultiHeadMLP
from core.dormancy import calculate_dormancy_scores
from agents.multihead_agent import MultiHeadDQNAgent
from core.env_wrapper import PadEnvWrapper


# ─────────────────────────────────────────────────────────────────
# TopK Sparse Autoencoder
# ─────────────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    """
    Sparse Autoencoder with TopK activation function.
    
    Architecture: x ∈ R^d  →  encoder  →  f ∈ R^m  (TopK sparse)  →  decoder  →  x̂ ∈ R^d
    
    References:
      - Bricken et al. 2023 (Anthropic)
      - Gao et al. 2024 (OpenAI) - TopK SAE
    """
    def __init__(self, d_input: int, d_hidden: int, k: int):
        """
        Args:
            d_input:  Input/output dimension (activation space, e.g. 256)
            d_hidden: Dictionary size (overcomplete, e.g. 1024 = 4x expansion)
            k:        Number of active features (TopK sparsity, e.g. 20-40)
        """
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.k = k

        # Encoder: input -> hidden pre-activations
        self.W_enc = nn.Linear(d_input, d_hidden, bias=True)

        # Decoder: sparse latents -> reconstruction
        # We keep decoder columns unit-norm (normalised dict atoms)
        self.W_dec = nn.Linear(d_hidden, d_input, bias=True)

        # Initialise decoder columns to unit norm
        with torch.no_grad():
            nn.init.orthogonal_(self.W_dec.weight)
            self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=0)

    def encode(self, x: torch.Tensor):
        """Returns sparse latent vector f in R^m with exactly k nonzeros."""
        pre_acts = self.W_enc(x)                      # (B, m)
        # TopK: keep only the k largest activations, zero the rest
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        f = torch.zeros_like(pre_acts)
        f.scatter_(-1, topk_idx, F.relu(topk_vals))   # ReLU so features are non-negative
        return f

    def decode(self, f: torch.Tensor):
        """Reconstruct activation from sparse latent."""
        return self.W_dec(f)

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    def normalise_decoder(self):
        """Project decoder columns back onto unit sphere after each optimiser step."""
        with torch.no_grad():
            self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=0)


# ─────────────────────────────────────────────────────────────────
# SAE Training
# ─────────────────────────────────────────────────────────────────

def train_sae(
    activations: torch.Tensor,
    d_input: int,
    expansion: int = 4,
    k: int = None,
    n_epochs: int = 50,
    batch_size: int = 512,
    lr: float = 2e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> TopKSAE:
    """
    Train a TopK SAE on a fixed dataset of activations.
    
    Loss = ||x - x_hat||_2^2  +  auxiliary_loss
    
    Auxiliary loss: penalises 'dead' latents that are never active over a batch,
    encouraging all dictionary atoms to be used (prevents index collapse).
    
    Args:
        activations: Tensor of shape (N, d_input) -- post-ReLU activations
        expansion:   d_hidden = expansion * d_input
        k:           TopK count. If None, defaults to expansion * 5 (i.e. 5% sparsity).
    """
    d_hidden = expansion * d_input
    if k is None:
        k = max(10, d_hidden // (expansion * 5))  # ~5% of hidden dim active

    sae = TopKSAE(d_input, d_hidden, k).to(device)
    optimiser = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)

    activations = activations.to(device)
    N = activations.shape[0]
    n_batches = max(1, N // batch_size)

    if verbose:
        print(f"\n[SAE] Training: d={d_input} -> m={d_hidden} | k={k} | N={N} | epochs={n_epochs}")
        print(f"[SAE] Sparsity target: {k}/{d_hidden} = {100*k/d_hidden:.1f}% active per sample")

    for epoch in range(1, n_epochs + 1):
        # Shuffle
        perm = torch.randperm(N, device=device)
        acts_shuffled = activations[perm]

        epoch_loss = 0.0
        epoch_recon = 0.0

        for i in range(n_batches):
            batch = acts_shuffled[i * batch_size: (i + 1) * batch_size]

            x_hat, f = sae(batch)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_hat, batch)

            # Auxiliary loss: fight index collapse (Anthropic ghost grads / Gao 2024).
            # For each dead latent (zero in this batch), we reconstruct using that latent
            # alone at the expected activation norm and compute the residual.
            # This creates a gradient signal to revive dead latents.
            with torch.no_grad():
                pre_acts = sae.W_enc(batch)        # (B, m)
                residual = batch - x_hat            # (B, d) -- what the SAE failed to explain
            is_dead = (f.sum(0) == 0)               # (m,) bool: never fired in this batch
            if is_dead.any():
                # Ghost gradient: project residual onto dead decoder directions
                # Encourages dead latents to explain the unexplained residual
                dead_dec_dirs = sae.W_dec.weight[:, is_dead]        # (d, n_dead)
                projections = residual @ dead_dec_dirs               # (B, n_dead)
                # Aux loss: push dead latents to fire proportionally to residual projection
                aux_target = F.relu(projections)                     # (B, n_dead) non-negative targets
                dead_pre_acts = pre_acts[:, is_dead]                 # (B, n_dead)
                aux_loss = 1.0 * F.mse_loss(dead_pre_acts, aux_target)
            else:
                aux_loss = torch.tensor(0.0, device=device)

            loss = recon_loss + aux_loss

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimiser.step()
            sae.normalise_decoder()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()

        scheduler.step()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            avg_loss  = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            # Compute sparsity and dead latent stats on last batch
            with torch.no_grad():
                _, f_test = sae(activations[:min(1000, N)])
                dead_pct = (f_test.sum(0) == 0).float().mean().item() * 100
                mean_l0  = (f_test > 0).float().sum(-1).mean().item()
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | recon={avg_recon:.4f} "
                  f"| dead_latents={dead_pct:.1f}% | mean_L0={mean_l0:.1f}")

    return sae


# ─────────────────────────────────────────────────────────────────
# Activation Collection
# ─────────────────────────────────────────────────────────────────

def collect_activations(agent: MultiHeadDQNAgent, layer_idx: int = -1, n_samples: int = 10000) -> torch.Tensor:
    """
    Collect post-ReLU activations from a specific hidden layer using the replay buffer.
    
    Args:
        layer_idx: which hidden layer to probe. -1 means last hidden layer.
        n_samples: number of activation vectors to collect
    """
    all_acts = []
    
    while sum(a.shape[0] for a in all_acts) < n_samples:
        sample_size = min(agent.batch_size, n_samples)
        if len(agent.memory) < sample_size:
            break
        states, _, _, _, _ = agent.memory.sample(sample_size)
        states_t = torch.FloatTensor(states)
        
        with torch.no_grad():
            _, activations = agent.network(states_t, return_activations=True, head_idx=0)
        
        # Grab the specified hidden layer
        target_layer = activations[layer_idx]  # (batch, d)
        all_acts.append(target_layer.cpu())
    
    return torch.cat(all_acts, dim=0)[:n_samples]


# ─────────────────────────────────────────────────────────────────
# Evaluation Utilities
# ─────────────────────────────────────────────────────────────────

def evaluate_standard(agent, env_id, n_episodes=20):
    """Baseline: all neurons active."""
    env = gym.make(env_id)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q = agent.network(state_t, head_idx=0)
            action = q.argmax(dim=1).item()
            state, r, term, trunc, _ = env.step(action)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    env.close()
    return np.mean(rewards), np.std(rewards)


def evaluate_coord_masked(agent, env_id, active_masks, n_episodes=20):
    """Coordinate-basis masking: zero out dormant neuron indices."""
    env = gym.make(env_id)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q = agent.network.forward_masked(state_t, active_masks, head_idx=0)
            action = q.argmax(dim=1).item()
            state, r, term, trunc, _ = env.step(action)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    env.close()
    return np.mean(rewards), np.std(rewards)


def evaluate_sae_masked(agent, env_id, sae: TopKSAE, layer_idx: int,
                        dead_latent_mask: torch.Tensor, n_episodes=20):
    """
    SAE-based feature masking:
      1. Run forward pass up to target layer
      2. Encode with SAE -> sparse latents f
      3. Zero out 'dead' SAE latents (those that never fire on Task A data)
      4. Decode SAE -> reconstruct layer activations x_hat
      5. Inject x_hat into the network as if it were the layer output
      6. Continue forward pass through remaining layers + output head
    
    dead_latent_mask: bool tensor of shape (d_hidden,) where True = dead latent to zero out
    """
    env = gym.make(env_id)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
    
    network = agent.network
    # Resolve layer_idx: support negative indexing
    n_layers = len(network.layers)
    probe_layer = layer_idx % n_layers  # e.g. -1 -> last hidden layer index

    sae.eval()
    rewards = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)  # (1, state_dim)
                
                # --- Partial forward up to probe_layer (inclusive) ---
                out = state_t
                for i, layer in enumerate(network.layers):
                    out = F.relu(layer(out))
                    if i == probe_layer:
                        # Intercept here: SAE encode -> mask dead -> decode
                        f = sae.encode(out)
                        # Mask out dead latents (latents that never fired on Task A data)
                        f = f * (~dead_latent_mask).float().unsqueeze(0)
                        out = sae.decode(f)
                        # NOTE: Do NOT apply ReLU here -- the SAE decoder already approximates
                        # the post-ReLU activation distribution. Adding another ReLU would
                        # systematically cut negative reconstruction residuals and distort Q-values.
                        # Clamp to non-negative to stay in the valid activation range.
                        out = out.clamp(min=0.0)
                
                # --- Continue through remaining hidden layers ---
                for i, layer in enumerate(network.layers):
                    if i > probe_layer:
                        out = F.relu(layer(out))
                
                # Output head
                q = network.output_layers[0](out)
            
            action = q.argmax(dim=1).item()
            state, r, term, trunc, _ = env.step(action)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    
    env.close()
    return np.mean(rewards), np.std(rewards)


def evaluate_sae_passthrough(agent, env_id, sae: TopKSAE, layer_idx: int, n_episodes=20):
    """
    SAE passthrough control (NO masking):
      encode -> decode all features without zeroing any latents.

    Isolates the pure cost of SAE reconstruction noise from the masking decision.
    If drop([D]) ~= drop([C]), the gap is reconstruction noise, not masking error.
    If drop([D]) << drop([C]), zeroing dead latents is removing real task features.
    """
    env = gym.make(env_id)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)

    network = agent.network
    n_layers = len(network.layers)
    probe_layer = layer_idx % n_layers

    sae.eval()
    rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)

                out = state_t
                for i, layer in enumerate(network.layers):
                    out = F.relu(layer(out))
                    if i == probe_layer:
                        # Full passthrough: encode -> decode, NO masking
                        f = sae.encode(out)
                        out = sae.decode(f)
                        out = out.clamp(min=0.0)

                for i, layer in enumerate(network.layers):
                    if i > probe_layer:
                        out = F.relu(layer(out))

                q = network.output_layers[0](out)

            action = q.argmax(dim=1).item()
            state, r, term, trunc, _ = env.step(action)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)

    env.close()
    return np.mean(rewards), np.std(rewards)


# ─────────────────────────────────────────────────────────────────
# Training + Pilot Evaluation
# ─────────────────────────────────────────────────────────────────

def train_to_convergence(agent, env_id, max_steps=150000, threshold=400.0):
    env = gym.make(env_id)
    env = PadEnvWrapper(env, max_state_dim=8, max_action_dim=4)
    state, _ = env.reset()
    
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = max_steps // 5

    ep_reward = 0.0
    ep_count = 0
    recent = deque(maxlen=10)

    for step in range(1, max_steps + 1):
        eps = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step / epsilon_decay)
        action = agent.select_action(state, eps)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        agent.step(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward

        if done:
            recent.append(ep_reward)
            ep_count += 1
            ep_reward = 0.0
            state, _ = env.reset()
            if len(recent) >= 10 and np.mean(recent) >= threshold:
                print(f"  Converged at step {step} (ep {ep_count}), mean10={np.mean(recent):.1f}")
                break

    env.close()


def run_pilot(seed=42, width=256, env_a="CartPole-v1",
              sae_expansion=4, sae_k=20, sae_epochs=100,
              n_collect=10000, n_eval_episodes=20):
    
    print("=" * 65)
    print(" SAE PILOT EXPERIMENT")
    print(f" Env: {env_a} | Width: {width} | Seed: {seed}")
    print(f" SAE: expansion={sae_expansion}x, k={sae_k}, epochs={sae_epochs}")
    print("=" * 65)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Phase 1: Train to convergence
    print("\n[Phase 1] Training CartPole agent...")
    agent = MultiHeadDQNAgent(
        state_dim=8, action_dim=4, hidden_dims=[width, width],
        replay_ratio=0.25, device="cpu", num_heads=1
    )
    agent.set_head(0)
    train_to_convergence(agent, env_a, max_steps=150000, threshold=400.0)

    # Collect activations from last hidden layer
    print(f"\n[Collect] Sampling {n_collect} post-ReLU activations (layer=-1, dim={width})...")
    acts = collect_activations(agent, layer_idx=-1, n_samples=n_collect)
    print(f"  Collected: {acts.shape}")

    # Coordinate-basis dormancy (existing method)
    print("\n[Dormancy] Computing coordinate-basis dormancy scores...")
    states_sample, _, _, _, _ = agent.memory.sample(agent.batch_size)
    states_t = torch.FloatTensor(states_sample)
    with torch.no_grad():
        _, all_activations = agent.network(states_t, return_activations=True, head_idx=0)
    dormant_indices, dormancy_pcts = calculate_dormancy_scores(all_activations, tau=0.025)
    n_dormant = [d.sum().item() for d in dormant_indices]
    print(f"  Dormancy: {[f'{p:.1f}%' for p in dormancy_pcts]} | nDormant/layer: {n_dormant}")
    active_masks = [~d for d in dormant_indices]

    # Train SAE
    sae = train_sae(
        acts, d_input=width, expansion=sae_expansion, k=sae_k,
        n_epochs=sae_epochs, batch_size=512, lr=2e-4, device="cpu", verbose=True,
    )
    sae.eval()

    # Identify dead SAE latents on Task A data
    print("\n[SAE] Identifying dead latents on Task A activation distribution...")
    with torch.no_grad():
        _, f_all = sae(acts)
    # Fire rate: fraction of samples in which each latent was active.
    # Using a small positive threshold (0.1%) rather than exact-zero to be robust:
    # a latent that fired in <0.1% of training samples is treated as 'dead'.
    fire_rate = (f_all > 0).float().mean(0)  # (d_hidden,) in [0, 1]
    dead_threshold = 0.001  # latent must fire in at least 0.1% of samples to be considered 'live'
    dead_latent_mask = (fire_rate < dead_threshold)  # True = dead
    n_dead  = dead_latent_mask.sum().item()
    n_total = dead_latent_mask.numel()
    pct_dead = 100.0 * n_dead / n_total
    print(f"  Dead latents (<{dead_threshold*100:.1f}% fire rate): {n_dead}/{n_total} ({pct_dead:.1f}%)")
    print(f"  Live latents: {n_total - n_dead}/{n_total} ({100-pct_dead:.1f}%)")
    print(f"  Fire rate stats: min={fire_rate.min():.4f}, mean={fire_rate.mean():.4f}, max={fire_rate.max():.4f}")

    # SAE Reconstruction quality
    with torch.no_grad():
        x_hat, _ = sae(acts)
    recon_err = F.mse_loss(x_hat, acts).item()
    recon_cos = F.cosine_similarity(x_hat, acts, dim=-1).mean().item()
    print(f"  SAE Recon MSE: {recon_err:.4f} | Cosine sim: {recon_cos:.4f}")

    # Evaluate: Three conditions
    print(f"\n[Eval] Running {n_eval_episodes}-episode evaluation under 3 conditions...")
    
    mean_all,  std_all  = evaluate_standard(agent, env_a, n_eval_episodes)
    print(f"  [A] Baseline (all neurons):      {mean_all:.1f} +/- {std_all:.1f}")

    mean_coord, std_coord = evaluate_coord_masked(agent, env_a, active_masks, n_eval_episodes)
    drop_coord = mean_all - mean_coord
    print(f"  [B] Coord-basis dormant masked:  {mean_coord:.1f} +/- {std_coord:.1f}  (drop={drop_coord:+.1f})")

    mean_sae, std_sae = evaluate_sae_masked(
        agent, env_a, sae, layer_idx=-1, dead_latent_mask=dead_latent_mask, n_episodes=n_eval_episodes
    )
    drop_sae = mean_all - mean_sae
    print(f"  [C] SAE feature masked:          {mean_sae:.1f} +/- {std_sae:.1f}  (drop={drop_sae:+.1f})")

    mean_pass, std_pass = evaluate_sae_passthrough(agent, env_a, sae, layer_idx=-1, n_episodes=n_eval_episodes)
    drop_pass = mean_all - mean_pass
    print(f"  [D] SAE passthrough (no mask):   {mean_pass:.1f} +/- {std_pass:.1f}  (drop={drop_pass:+.1f})")

    # Summary Table
    print("\n" + "=" * 65)
    print(" RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'Condition':<35} {'Mean':>8} {'Std':>8} {'Drop':>8}")
    print("-" * 65)
    print(f"{'[A] All neurons (baseline)':<35} {mean_all:>8.1f} {std_all:>8.1f} {'--':>8}")
    print(f"{'[B] Coord-basis (dormant masked)':<35} {mean_coord:>8.1f} {std_coord:>8.1f} {drop_coord:>+8.1f}")
    print(f"{'[C] SAE feature masked':<35} {mean_sae:>8.1f} {std_sae:>8.1f} {drop_sae:>+8.1f}")
    print(f"{'[D] SAE passthrough (no mask)':<35} {mean_pass:>8.1f} {std_pass:>8.1f} {drop_pass:>+8.1f}")
    print("=" * 65)

    # -- Decompose the SAE masking drop into reconstruction noise vs masking cost --
    # drop([C]) = recon_noise + masking_cost
    # drop([D]) = recon_noise (passthrough has no masking)
    # masking_cost = drop([C]) - drop([D])
    recon_noise_cost = drop_pass
    masking_cost     = drop_sae - drop_pass

    print("\n[Drop Decomposition]")
    print(f"  drop([C] SAE masked)     = {drop_sae:+.1f}")
    print(f"  drop([D] passthrough)    = {drop_pass:+.1f}  <- pure reconstruction noise")
    print(f"  masking_cost             = {masking_cost:+.1f}  <- extra cost from zeroing dead latents")
    print()
    if abs(masking_cost) < 10:
        print("  => Masking cost is NEGLIGIBLE. The dead latents genuinely carry no")
        print("     useful task information. SAE masking is principled -- the drop")
        print("     is driven by reconstruction noise, not incorrect feature removal.")
    elif masking_cost > 0:
        print("  => Masking cost is POSITIVE: zeroing dead latents IS removing live info.")
        print("     The dead latent identification threshold may be too aggressive.")
    else:
        print("  => Masking cost is NEGATIVE: zeroing dead latents HELPS performance.")
        print("     This would be extraordinary -- dead latents were adding noise.")

    print()
    if abs(drop_coord) < abs(drop_pass):
        print("  => Coord-basis masking is BETTER than even pure SAE passthrough.")
        print("     This means the reconstruction noise from the SAE is the primary")
        print("     bottleneck. The SAE approach needs better reconstruction fidelity")
        print("     before it can beat the simple coordinate-basis approach.")
    else:
        print("  => SAE passthrough is competitive with or better than coord-basis.")
        print("     With better masking, the SAE approach should outperform.")
    
    print("\n[Interpretation]")
    if abs(drop_sae) < abs(drop_coord):
        advantage = abs(drop_coord) - abs(drop_sae)
        print(f"  OK SAE masking preserves performance BETTER than coord-basis masking.")
        print(f"  OK Performance advantage of SAE masking: {advantage:.1f} pts")
        print(f"  OK This supports the superposition hypothesis: dormant neurons ARE")
        print(f"     entangled with active task features in coordinate space, but SAE")
        print(f"     features correctly disentangle them.")
    elif abs(drop_sae) > abs(drop_coord):
        disadvantage = abs(drop_sae) - abs(drop_coord)
        print(f"  XX Coord-basis masking is surprisingly better ({disadvantage:.1f} pts).")
        print(f"  XX Possible causes: SAE reconstruction error corrupts Q-value decisions,")
        print(f"     or dormant neurons truly carry near-zero distributed activation mass.")
    else:
        print(f"  ~~ Both methods produce similar drops. SAE may need more training or")
        print(f"     higher expansion factor to reveal cleaner feature decomposition.")
    
    print(f"\n  SAE stats: {n_dead}/{n_total} dead latents ({pct_dead:.1f}%) vs"
          f" {n_dormant[1]}/{width} dead neurons ({dormancy_pcts[1]:.1f}%)")
    print(f"  SAE reconstruction quality: MSE={recon_err:.4f}, cos_sim={recon_cos:.4f}")
    print("=" * 65)

    return {
        "baseline": (mean_all, std_all),
        "coord_masked": (mean_coord, std_coord),
        "sae_masked": (mean_sae, std_sae),
        "drop_coord": drop_coord,
        "drop_sae": drop_sae,
        "sae_dead_pct": pct_dead,
        "neuron_dead_pct": np.mean(dormancy_pcts),
        "recon_mse": recon_err,
        "recon_cos": recon_cos,
    }


if __name__ == "__main__":
    results = run_pilot(
        seed=42,
        width=256,
        env_a="CartPole-v1",
        sae_expansion=4,    # 256 -> 1024 latents
        sae_k=32,           # 32/1024 = 3.1% active per sample (slightly more than v1's 2%)
        sae_epochs=200,     # more epochs to recover from index collapse
        n_collect=10000,    # activation vectors to fit SAE on
        n_eval_episodes=20,
    )
