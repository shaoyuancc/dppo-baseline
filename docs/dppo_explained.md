# DPPO: Diffusion Policy Policy Optimization - Technical Explanation

This document explains the key technical details of DPPO, specifically:

1. How log probabilities are calculated
2. How the mean μ differs between DDPM and DDIM
3. What oldlogprobs and newlogprobs represent, and whether they match
4. How gradients flow through the computation
5. The `reward_horizon` parameter
6. The role of `min_logprob_denoising_std` and `min_sampling_denoising_std`
7. GAE (Generalized Advantage Estimation) and the bias-variance trade-off
8. Key training metrics (ratio, explained variance)
9. Why reducing `act_steps` can increase explained variance

---

## Part 1: Log Probability Calculation

### The Denoising Chain as an MDP

DPPO treats the diffusion denoising process as a Markov Decision Process (MDP). During sampling, we generate a chain of intermediate states:

```
x_K (pure noise) → x_{K-1} → ... → x_1 → x_0 (final action)
```

Each denoising transition `p(x_{t-1} | x_t)` is modeled as a **Gaussian distribution**:

$$p(x_{t-1} | x_t, s) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, s), \sigma_t^2 I)$$

where:
- **μ_θ** is the mean, computed from the neural network's noise prediction
- **σ_t** is the standard deviation, derived from the diffusion schedule (details below)
- **s** is the conditioning observation

### Key Insight: Variance is Fixed, Not Learned

**The variance σ_t² comes from the pre-defined diffusion schedule, NOT from the neural network.** The network only predicts noise ε, which determines the mean μ. The variance is entirely determined by pre-computed schedule coefficients.

For DDPM, `logvar` comes directly from the schedule:

```python
# In diffusion.py __init__():
# β̃_t = σ_t² = β_t * (1-α̅_{t-1}) / (1-α̅_t)
self.ddpm_var = (
    self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
)
self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))
```

For DDIM, `logvar` comes from the eta model and the alpha schedule:

```python
# sigma = eta * sqrt((1-alpha_prev)/(1-alpha) * (1 - alpha/alpha_prev))
sigma = (
    etas
    * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)) ** 0.5
).clamp_(min=1e-10)
var = sigma**2
logvar = torch.log(var)
```

In both cases, the network's noise prediction does not affect the variance. The only way the variance changes is through the schedule (DDPM) or through the eta model (DDIM, which is a separate small network or fixed parameter, not the main actor).

---

## Part 2: How the Mean (μ) is Computed - DDPM vs DDIM

**No, μ is NOT computed the same way for DDPM and DDIM.** They use different formulas, though both start from the same noise prediction.

### DDPM Mean Computation

**Step 1**: Reconstruct x_0 from noise prediction:

```python
# x₀ = √(1/α̅_t) * x_t - √(1/α̅_t - 1) * ε
x_recon = (
    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
)
```

**Step 2**: Compute the DDPM posterior mean:

```python
# μ_t = coef1 * x₀ + coef2 * x_t
# where coef1 = β_t * √α̅_{t-1} / (1-α̅_t)
#       coef2 = √α_t * (1-α̅_{t-1}) / (1-α̅_t)
mu = (
    extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
    + extract(self.ddpm_mu_coef2, t, x.shape) * x
)
```

This is a weighted combination of the predicted clean sample x₀ and the current noisy sample x_t.

### DDIM Mean Computation

**Step 1**: Reconstruct x_0 from noise prediction (different formula):

```python
# x₀ = (x_t - √(1-α_t) * ε) / √α_t
alpha = extract(self.ddim_alphas, index, x.shape)
sqrt_one_minus_alpha = extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)
x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
```

**Step 2**: Compute the DDIM mean:

```python
# μ = √α_{t-1} * x₀ + √(1-α_{t-1} - σ²) * ε
sigma = (
    etas * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)) ** 0.5
).clamp_(min=1e-10)
dir_xt_coef = (1.0 - alpha_prev - sigma**2).clamp_(min=0).sqrt()
mu = (alpha_prev**0.5) * x_recon + dir_xt_coef * noise
```

### Key Differences

| Aspect | DDPM | DDIM |
|--------|------|------|
| **x₀ reconstruction** | Uses `sqrt_recip_alphas_cumprod` (indexed by raw timestep `t`) | Uses `ddim_alphas` (indexed by DDIM step index) |
| **Mean formula** | Weighted sum of x₀ and x_t | Weighted sum of x₀ and ε (the noise) |
| **Variance source** | Fixed by schedule: β̃_t | Controlled by eta model: η * f(alphas) |
| **Number of steps** | Uses all denoising steps (e.g., 100) | Uses subsampled steps (e.g., 5 or 10) |

Despite these differences, both ultimately compute μ as a **deterministic function of the noise prediction ε**, the current noisy sample x_t, and pre-computed schedule coefficients.

---

## Part 3: oldlogprobs vs newlogprobs - What They Represent

### What Are They?

**`oldlogprobs`**: The log probability of each denoising transition, computed **immediately after rollout** using the policy weights that generated the data.

**`newlogprobs`**: The log probability of the **same** denoising transitions, recomputed **during PPO updates** using the current (possibly updated) policy weights.

### The Timeline

```
┌─────────────────── ROLLOUT PHASE ───────────────────┐
│                                                       │
│  1. Use current policy (actor_ft) to sample actions   │
│     x_K → x_{K-1} → ... → x_0                       │
│     Store the full chain in buffer                    │
│                                                       │
│  2. Immediately after rollout, compute oldlogprobs:   │
│     For each pair (x_t, x_{t-1}) in the chain:       │
│       μ = p_mean_var(x_t, t, obs)                    │
│       oldlogprob = Normal(μ, σ).log_prob(x_{t-1})    │
│     Store oldlogprobs in buffer                       │
│                                                       │
│  (Policy weights are frozen at this point)            │
└───────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────── PPO UPDATE PHASE ────────────────┐
│                                                       │
│  For each update epoch:                               │
│    For each mini-batch:                               │
│      3. Recompute newlogprobs with CURRENT weights:   │
│         μ_new = p_mean_var(x_t, t, obs)  ← UPDATED  │
│         newlogprob = Normal(μ_new, σ).log_prob(x_t-1)│
│                                                       │
│      4. Compute ratio = exp(newlogprobs - oldlogprobs)│
│      5. Compute PPO loss and update weights           │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### Will newlogprobs Equal oldlogprobs?

**At the very first mini-batch of the very first epoch**: Yes, `newlogprobs ≈ oldlogprobs` (ratio ≈ 1.0), because the network weights haven't been updated yet since the rollout.

**After weight updates**: No, `newlogprobs ≠ oldlogprobs`. The network has been updated, so `p_mean_var` produces a different mean μ_new, which leads to a different log probability. This is precisely what the PPO ratio measures - how much the policy has changed.

### Does the Recomputed Mean Match the Rollout Mean?

**Before any weight update**: The recomputed mean will be **exactly the same** as during rollout, because:
- The same `chains_prev` (x_t) values are fed in
- The same observation `cond` is fed in
- The same network weights are used
- The mean is a deterministic function of these inputs

**After weight updates**: The mean will be **different**, because the actor_ft network now has different weights, so it predicts different noise ε, leading to a different μ.

**Important**: The actual sampled action x_{t-1} (stored as `chains_next`) is **never resampled**. It's the same tensor from the rollout. Only the mean μ (and therefore the log probability of that fixed action) changes as the network is updated.

### Code Walkthrough

Here's how oldlogprobs are computed (from `train_ppo_diffusion_truck2d_agent.py`):

```python
# After rollout, with no_grad():
logprobs = self.model.get_logprobs(obs, chains).cpu().numpy()
logprobs_trajs = ...  # Shape: (n_steps*n_envs, ft_denoising_steps, horizon_steps, action_dim)
```

Then during PPO updates, subsample pairs are passed to the loss function:

```python
# Per-batch during PPO update:
chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]    # x_t
chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]  # x_{t-1}
logprobs_b = logprobs_k[batch_inds_b, denoising_inds_b]     # oldlogprobs for this (x_t, x_{t-1}) pair

# Inside model.loss():
newlogprobs, eta = self.get_logprobs_subsample(
    obs, chains_prev, chains_next, denoising_inds, get_ent=True
)
# newlogprobs is recomputed with current weights
# oldlogprobs (logprobs_b) was computed right after rollout
```

---

## Part 4: The Gradient Chain - Detailed Explanation

### What "Gradient Chain" Means

In deep learning, we train networks by computing gradients of a loss function with respect to network parameters via backpropagation. The "gradient chain" describes **the path through which gradients flow backward** from the loss to the trainable parameters, following the chain rule of calculus.

For DPPO, we need gradients of the PPO loss to reach the actor network weights so they can be updated.

### Step-by-Step Gradient Flow

Here is the complete chain, annotated with what each step contributes:

**Step 1: PPO Loss → Ratio**

```python
pg_loss = torch.max(-advantages * ratio, -advantages * clipped_ratio).mean()
```

The gradient of the loss with respect to ratio depends on whether the sample was clipped. For unclipped samples:

$$\frac{\partial \text{pg\_loss}}{\partial \text{ratio}} = -\text{advantage} / \text{batch\_size}$$

**Step 2: Ratio → Log Ratio**

```python
ratio = logratio.exp()
```

Since ratio = exp(logratio), by the chain rule:

$$\frac{\partial \text{ratio}}{\partial \text{logratio}} = \text{ratio}$$

**Step 3: Log Ratio → newlogprobs**

```python
logratio = newlogprobs - oldlogprobs
```

`oldlogprobs` is a detached constant (no gradient), so:

$$\frac{\partial \text{logratio}}{\partial \text{newlogprobs}} = 1$$

**Step 4: newlogprobs → log_prob (per-element)**

```python
newlogprobs = log_prob.mean(dim=(-1, -2))  # Average over action dim and horizon
```

The gradient of mean distributes equally across all elements.

**Step 5: log_prob → μ (the critical step)**

```python
dist = Normal(mu, std)
log_prob = dist.log_prob(chains_next)
```

This is the Gaussian log probability: log p(x) = -0.5 * [log(2π) + log(σ²) + (x-μ)²/σ²]

The gradient with respect to μ:

$$\frac{\partial \log p(x)}{\partial \mu} = \frac{x - \mu}{\sigma^2}$$

**σ is detached from the computation graph** (it's either from the fixed schedule, or clipped to a constant). So the gradient only flows through μ.

Note that `chains_next` (the x in the formula) is also detached - it's a fixed tensor from the rollout buffer.

**Step 6: μ → noise_pred (through x_recon)**

For DDPM:
```python
x_recon = sqrt_recip * chains_prev - sqrt_recipm1 * noise_pred
mu = coef1 * x_recon + coef2 * chains_prev
```

Since `chains_prev` is detached and the coefficients are constants:

$$\frac{\partial \mu}{\partial \text{noise\_pred}} = -\text{coef1} \times \text{sqrt\_recipm1}$$

**Step 7: noise_pred → actor_ft.parameters()**

```python
noise_pred = actor_ft(chains_prev, t, cond=obs)
```

This is a standard neural network forward pass. Backpropagation computes gradients of `noise_pred` with respect to all network weights through the standard chain rule applied to each layer.

### Summary: What Gets Gradient and What Doesn't

| Tensor | Has gradient? | Why? |
|--------|--------------|------|
| `chains_prev` (x_t) | No | From rollout buffer, detached |
| `chains_next` (x_{t-1}) | No | From rollout buffer, detached |
| `oldlogprobs` | No | Computed after rollout, detached |
| `advantages` | No | Computed from rewards/values, detached |
| `sigma / logvar` | No | From fixed schedule or clipped constant |
| **`noise_pred`** | **Yes** | Output of actor_ft network |
| **`x_recon`** | **Yes** | Depends on noise_pred |
| **`mu`** | **Yes** | Depends on x_recon |
| **`newlogprobs`** | **Yes** | Depends on mu |
| **`ratio`** | **Yes** | Depends on newlogprobs |
| **`pg_loss`** | **Yes** | Depends on ratio |

### ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GRADIENT FLOW DIAGRAM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  pg_loss ◄────── PPO clipped objective                                  │
│     │                                                                   │
│     ▼                                                                   │
│   ratio ◄─────── exp(newlogprobs - oldlogprobs)                         │
│     │                    │              │                               │
│     │                    ▼              ▼                               │
│     │              newlogprobs    oldlogprobs (DETACHED, no gradient)   │
│     │                    │                                              │
│     │                    ▼                                              │
│     │              log_prob(chains_next)                                │
│     │                    │         │                                    │
│     │                    ▼         ▼                                    │
│     │                   mu    chains_next (DETACHED)                    │
│     │                    │                                              │
│     │                    ▼                                              │
│     │               x_recon = f(noise_pred, chains_prev)                │
│     │                    │              │                               │
│     │                    ▼              ▼                               │
│     │              noise_pred    chains_prev (DETACHED)                 │
│     │                    │                                              │
│     │                    ▼                                              │
│     │         actor_ft(chains_prev, t, obs)                             │
│     │                    │                                              │
│     ▼                    ▼                                              │
│  GRADIENTS ────────► actor_ft.parameters() ◄─── TRAINABLE WEIGHTS       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 5: The `reward_horizon` Parameter

### What It Is

`reward_horizon` controls **how many action steps (out of the full `horizon_steps`) contribute to the policy gradient**. It defaults to `act_steps`.

From `train_ppo_diffusion_agent.py`:
```python
# Reward horizon --- always set to act_steps for now
self.reward_horizon = cfg.get("reward_horizon", self.act_steps)
```

### Where It's Applied

In the loss function (`diffusion_ppo.py`):

```python
# newlogprobs shape: (B, Ta, Da) where Ta = horizon_steps
# Slice to only the first reward_horizon action steps:
newlogprobs = newlogprobs[:, :reward_horizon, :]
oldlogprobs = oldlogprobs[:, :reward_horizon, :]

# Then average over the kept dimensions:
newlogprobs = newlogprobs.mean(dim=(-1, -2)).view(-1)
oldlogprobs = oldlogprobs.mean(dim=(-1, -2)).view(-1)
```

### Why It Matters

The diffusion model predicts a full **action chunk** of `horizon_steps` actions (e.g., 32 steps). But typically only the first `act_steps` (e.g., 4 or 20) are actually **executed** in the environment before the next observation is taken.

```
Predicted chunk:  [a_0, a_1, a_2, a_3, ..., a_{31}]  (horizon_steps=32)
Actually executed: [a_0, a_1, a_2, a_3]                (act_steps=4)
Discarded:                                [a_4, ..., a_{31}]
```

The `reward_horizon` parameter (defaulting to `act_steps`) ensures that **gradients only flow through the action steps that actually affected the environment reward**. Actions that were predicted but never executed should not receive gradient signal, because:

1. Their quality was never tested in the environment
2. Pushing gradients through them adds noise to the update
3. The reward we observed only reflects the first `act_steps` actions

### Impact of Different Values

| `reward_horizon` | Effect |
|-------------------|--------|
| **= act_steps** (default) | Only train on actions that were actually executed. Clean gradient signal. |
| **< act_steps** | Only train on a subset of executed actions. Could be useful if later actions in the chunk are less reliable. |
| **> act_steps** | Also train on actions that were never executed. Noisy gradients since reward doesn't reflect these actions. |
| **= horizon_steps** | Train on ALL predicted actions including unexecuted ones. Maximum gradient noise. |

### Concrete Example

With `horizon_steps=32`, `act_steps=20`:

```python
# Log probs have shape (B, 32, action_dim) - one per predicted action step
newlogprobs = newlogprobs[:, :20, :]  # Keep only first 20 (actually executed)
# These 20 log probs get averaged together to form one scalar per batch element
newlogprobs = newlogprobs.mean(dim=(-1, -2))  # → (B,)
```

The policy gradient will update the network to make the first 20 actions more likely (if advantage is positive) or less likely (if negative), based on the actual reward observed. The remaining 12 predicted but unexecuted actions don't contribute to the gradient.

---

## Part 6: `min_logprob_denoising_std` Parameter

### Where It's Used

This parameter is applied **only during training** when computing log probabilities:

```python
# From get_logprobs() in diffusion_vpg.py
std = torch.exp(0.5 * logvar)
std = torch.clip(std, min=self.min_logprob_denoising_std)  # <-- HERE
dist = Normal(next_mean, std)
log_prob = dist.log_prob(chains_next)
```

### Why It's Needed

Recall the gradient of log probability w.r.t. mean:

$$\frac{\partial \log p(x)}{\partial \mu} = \frac{x - \mu}{\sigma^2}$$

**Problem**: When σ is very small (as it is at later denoising steps where the schedule variance approaches zero):
- Small deviations `(x - μ)` produce **huge** gradients because of the 1/σ² term
- Gradients can explode → training instability
- Log probabilities can become extremely negative

**Solution**: Clamp σ to a minimum value to cap the gradient magnitude.

### Impact of Different Values

| `min_logprob_denoising_std` | Effect |
|----------------------------|--------|
| **Higher (e.g., 0.1)** | Smaller gradients, smoother optimization landscape, more stable but slower learning |
| **Lower (e.g., 0.01)** | Larger gradients, sharper probability peaks, faster learning but potentially unstable |

Think of it as a **temperature** for the gradient computation:
- **Higher values** = "softer" policy, more forgiving of deviations from mean
- **Lower values** = "sharper" policy, strongly penalizes deviations from mean

### Practical Settings from Codebase

```yaml
# Robomimic tasks (end-effector control, simpler dynamics)
min_logprob_denoising_std: 0.1

# Truck 2D tasks (joint angle control, need faster learning)  
min_logprob_denoising_std: 0.01

# Reach subtask (precise control needed, but keep training stable)
min_logprob_denoising_std: 0.1  # Even when sampling std is 0.01
```

### Critical: The σ_logprob >> σ_sampling Mismatch and Jitter

When `min_logprob_denoising_std` is much larger than `min_sampling_denoising_std`, it **blinds PPO's safety mechanisms** and allows the policy to accumulate jitter over training iterations. This is especially damaging for joint-angle control where noise in each joint compounds through the kinematic chain.

#### The mechanism

PPO constrains policy changes via the importance sampling ratio:

```python
ratio = exp(newlogprobs - oldlogprobs)
# Clipped to [1-ε, 1+ε] to prevent destructive updates
```

The ratio measures "how much has the policy changed?" But this measurement is filtered through σ_logprob. When the mean μ shifts by δ, the change in log probability is approximately:

$$\Delta \log p \approx \frac{(x - \mu) \cdot \delta}{\sigma_{\text{logprob}}^2}$$

**When σ_logprob >> σ_sampling (e.g., 0.1 vs 0.01):**
- Large σ²_logprob in the denominator → Δlog_prob is tiny
- Even substantial shifts in μ barely change the ratio
- ratio stays ≈ 1.0 no matter how much the policy actually changes
- PPO clipping never activates → the safety constraint is blind
- `target_kl` early stopping never triggers (approx_kl stays artificially low)
- The policy drifts unconstrained each iteration

**When σ_logprob ≈ σ_sampling (both small, e.g., both 0.01):**
- Small σ²_logprob → Δlog_prob is large even for small shifts in μ
- The ratio quickly deviates from 1.0 when the policy changes
- PPO clipping activates early, constraining the update
- `target_kl` triggers appropriately, stopping updates before over-shooting

#### How jitter accumulates without proper constraints

Even with low sampling noise (smooth rollouts), the policy develops jitter through unconstrained drift:

1. Each PPO update slightly shifts the network's noise predictions
2. With large σ_logprob, the ratio stays ≈ 1.0 → clipping doesn't constrain these shifts
3. Small inconsistencies in gradient direction across mini-batches push predictions in slightly different directions for nearby inputs
4. Over many iterations, these unconstrained micro-shifts compound into high-frequency oscillations in the predicted noise
5. The mean μ itself becomes jittery → even deterministic sampling produces jerky actions

#### Why higher approx_kl is actually a good sign

When both σ values are lowered together, you may observe **higher approx_kl**. This is the signature of properly calibrated safety constraints: policy changes are now **visible** to the KL metric, so `target_kl` can stop updates before the policy drifts too far. With large σ_logprob, the KL was artificially suppressed, hiding real policy changes from the safety mechanism.

#### Joint-angle control makes this worse

For robomimic tasks (end-effector pose delta control), noise in each dimension is relatively independent -- jitter in x doesn't affect y. For joint-angle control (like truck unloading), noise in each joint compounds through the kinematic chain: small angular errors at the shoulder become large positional errors at the gripper. The same `min_sampling_denoising_std` value produces much more visible jitter in joint space than in end-effector space, and that jitter is more damaging to task performance (grasping, avoiding collisions, precise drops).

#### Practical recommendation for joint-angle control

Keep `min_logprob_denoising_std` close to or equal to `min_sampling_denoising_std`:

```yaml
# GOOD: matched values, PPO constraints work properly
min_sampling_denoising_std: 0.01
min_logprob_denoising_std: 0.01

# BAD: mismatch blinds PPO clipping, jitter accumulates
min_sampling_denoising_std: 0.01
min_logprob_denoising_std: 0.1
```

---

## Part 7: `min_sampling_denoising_std` for DDPM and DDIM

This parameter controls **exploration** during action sampling (inference/rollout), NOT during training.

### For DDPM Sampling

In DDPM, each denoising step samples:

$$x_{t-1} = \mu_t + \sigma_t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

The `min_sampling_denoising_std` sets a floor on σ_t:

```python
# From forward() in diffusion_vpg.py
if self.use_ddim:
    ...
else:
    # DDPM handling
    if deterministic and t == 0:
        std = torch.zeros_like(std)  # No noise at final step
    elif deterministic:
        std = torch.clip(std, min=1e-3)  # Small noise for stability
    else:
        std = torch.clip(std, min=min_sampling_denoising_std)  # Exploration noise
```

**DDPM behavior**:
- At `t=0`: Always deterministic (no noise added)
- At `t>0` with `deterministic=False`: Noise is at least `min_sampling_denoising_std`
- At `t>0` with `deterministic=True`: Minimal noise (1e-3) for numerical stability

### For DDIM Sampling

DDIM is **deterministic by default** (when eta=0). The stochasticity comes from:

1. **Eta (η) model**: Controls DDIM variance through:
   $$\sigma = \eta \cdot \sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t} \cdot (1 - \frac{\alpha_t}{\alpha_{t-1}})}$$

2. **`min_sampling_denoising_std`**: Provides a floor on the final std:

```python
if self.use_ddim:
    if deterministic:
        std = torch.zeros_like(std)  # Fully deterministic
    else:
        std = torch.clip(std, min=min_sampling_denoising_std)  # Add exploration
```

### Comparison: DDPM vs DDIM

| Aspect | DDPM | DDIM |
|--------|------|------|
| **Default behavior** | Stochastic (except t=0) | Deterministic (eta=0) |
| **min_sampling_std role** | Floor on schedule variance | Floor on eta-derived variance |
| **Typical steps** | Many (e.g., 100) | Few (e.g., 5 or 10) |
| **Exploration source** | Schedule variance | Eta model |

### Impact of Different Values

| `min_sampling_denoising_std` | DDPM Effect | DDIM Effect |
|------------------------------|-------------|-------------|
| **Higher (0.1)** | More exploration, diverse trajectories | Forces stochasticity even with eta=0 |
| **Lower (0.01)** | Closer to learned distribution | Near-deterministic with low eta |
| **0.0** | Use exact schedule variance | Fully deterministic |

**Key insight**: `min_sampling_denoising_std` and `min_logprob_denoising_std` serve different purposes and can be set independently:
- `min_sampling_denoising_std` = **exploration** during rollout
- `min_logprob_denoising_std` = **training stability** during optimization

---

## Part 8: GAE (Generalized Advantage Estimation) Explained

### The Fundamental Definition of Advantage

The advantage function measures **how much better a specific action is compared to the average action from that state**:

$$A(s, a) = Q(s, a) - V(s)$$

where:
- **Q(s, a)**: the expected total return from taking action a in state s, then following the policy thereafter
- **V(s)**: the expected total return from state s when following the policy (averaged over all actions)

If A > 0, the action was **better than average**. If A < 0, it was **worse than average**.

### The Problem: We Don't Have Q

In PPO, we don't learn a Q function. We only have:
- A learned value function V(s) (the critic)
- Observed rewards from rollouts

So we need to **estimate** the advantage from these ingredients. There are multiple ways to do this, and GAE is a method that lets you trade off between them.

### Building Intuition: Two Extreme Estimators

**Estimator 1: One-step TD residual (low variance, high bias)**

$$\hat{A}_t^{(1)} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

This says: "the advantage of what I did is the reward I got, plus the discounted value of where I ended up, minus the value of where I started."

This is **low variance** because it only depends on one reward sample r_t. But it's **high bias** because it relies on V(s_{t+1}) being accurate -- if the critic is wrong, this estimate is wrong.

**Estimator 2: Full Monte Carlo return (high variance, zero bias)**

$$\hat{A}_t^{(\infty)} = \sum_{l=0}^{T-t} \gamma^l r_{t+l} - V(s_t) = G_t - V(s_t)$$

This says: "the advantage is the actual total discounted reward I observed minus what the critic predicted."

This is **zero bias** because it uses the actual observed return G_t. But it's **high variance** because G_t depends on every reward from now until the episode ends -- lots of stochasticity accumulates.

**The connection to A = Q - V**: The Monte Carlo return G_t is an unbiased single-sample estimate of Q(s_t, a_t). So A_t^(∞) = G_t - V(s_t) is directly the A = Q - V definition, using an unbiased estimate of Q.

### GAE: Interpolating Between the Two

GAE introduces a parameter λ ∈ [0, 1] that smoothly interpolates:

$$\hat{A}_t^{GAE} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$$

where each δ_{t+l} = r_{t+l} + γV(s_{t+l+1}) - V(s_{t+l}) is a one-step TD residual.

Expanding this:
- **λ = 0**: A_t = δ_t (one-step TD, low variance, high bias)
- **λ = 1**: A_t = Σ γ^l δ_{t+l} = G_t - V(s_t) (Monte Carlo, high variance, zero bias)
- **λ ∈ (0, 1)**: A weighted combination (the typical setting is λ ≈ 0.95)

### How GAE Connects to A = Q - V

The fundamental definition A = Q - V still holds conceptually. GAE provides a **biased estimate** of Q - V that has **controllable variance**:

$$\hat{A}_t^{GAE} \approx Q(s_t, a_t) - V(s_t)$$

When λ = 1, this is exactly Q - V with Q estimated by the Monte Carlo return (unbiased but noisy). When λ < 1, we trade some bias for lower variance by partially trusting the critic's bootstrapped estimates V(s_{t+l}) instead of waiting for all future rewards.

The intuition: instead of asking "what was the actual total reward?" (Monte Carlo), GAE asks "what was the actual reward for a few steps, then what does the critic think will happen after that?" -- with λ controlling how many steps of actual reward we use before handing off to the critic.

### The Code

From `train_ppo_diffusion_agent.py`:

```python
advantages_trajs = np.zeros_like(reward_trajs)
lastgaelam = 0
for t in reversed(range(self.n_steps)):
    if t == self.n_steps - 1:
        nextvalues = self.model.critic(obs_venv_ts).reshape(1, -1).cpu().numpy()
    else:
        nextvalues = values_trajs[t + 1]
    nonterminal = 1.0 - terminated_trajs[t]
    # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    delta = (
        reward_trajs[t] * self.reward_scale_const
        + self.gamma * nextvalues * nonterminal
        - values_trajs[t]
    )
    # A_t = δ_t + γλ * A_{t+1}  (recursive form of the GAE sum)
    advantages_trajs[t] = lastgaelam = (
        delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
    )
# R_t = A_t + V(s_t)  (the GAE return target for the critic)
returns_trajs = advantages_trajs + values_trajs
```

Note the recursive form: `A_t = δ_t + γλ * A_{t+1}`. This is mathematically equivalent to the sum Σ(γλ)^l δ_{t+l}, just computed efficiently from the end of the trajectory backwards.

The **returns** R_t = A_t + V(s_t) are what the critic is trained to predict, and what explained variance is measured against.

### Important: Each "Step" t Is One Action Chunk

As noted in the code comment: `gamma here is applied to reward every act_steps, instead of every env step`. Each step t in the GAE computation corresponds to one action chunk of `act_steps` environment steps. The reward r_t is the **sum** of all environment step rewards within that chunk (done by the `multi_step.py` wrapper). The observation s_t is taken at the start of each chunk.

### Impact of Advantage Variance on Training

The advantage estimate directly multiplies the policy gradient. In PPO:

```python
pg_loss1 = -advantages * ratio
pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
pg_loss = torch.max(pg_loss1, pg_loss2).mean()
```

The gradient that updates the actor is proportional to: advantage × ∂log_prob/∂θ. If the advantage estimates are noisy, the gradient signal is noisy. This has direct consequences:

**Low variance advantages (good critic, small act_steps, appropriate λ):**

1. **More stable training**: Each gradient step pushes the policy in a consistent direction. Consecutive batches agree on which actions are good/bad.
2. **Better sample efficiency**: Each rollout produces a cleaner learning signal. Fewer samples are needed to make progress because the gradient direction is accurate.
3. **Faster convergence**: The policy can take larger effective steps without overshooting, because the gradient direction is reliable.
4. **Less sensitivity to hyperparameters**: With clean gradients, training is more robust to learning rate, batch size, etc.

**High variance advantages (poor critic, large act_steps, λ too high):**

1. **Unstable training**: Gradients are noisy -- one batch says "action X was great," the next says "action X was terrible." The policy oscillates or fails to converge.
2. **Poor sample efficiency**: Many rollouts are needed to average out the noise and make progress. A lot of data is "wasted" on contradictory gradient signals.
3. **Requires conservative hyperparameters**: Must use smaller learning rates, more clipping, larger batches, etc., to compensate for noise. This slows training further.
4. **Can cause policy collapse**: In severe cases, a string of high-variance gradient updates can push the policy far from the pre-trained initialization, destroying learned behaviors.

### The Vicious/Virtuous Cycle

There's a feedback loop between advantage quality and critic quality:

**Virtuous cycle (low variance advantages):**
```
Good critic → accurate advantages → clean policy gradients → policy improves predictably
    → critic's predictions stay accurate → advantages stay clean → ...
```

**Vicious cycle (high variance advantages):**
```
Poor critic → noisy advantages → noisy policy gradients → policy changes unpredictably
    → critic's predictions become even less accurate → advantages get noisier → ...
```

This is why `n_critic_warmup_itr` matters: warming up the critic before allowing actor updates ensures the virtuous cycle starts from a reasonable baseline.

### Practical Implications for DPPO

| Knob | Effect on Advantage Variance |
|------|------------------------------|
| **Smaller `act_steps`** | Lower (less reward noise per chunk) |
| **Higher `gae_lambda`** (closer to 1) | Higher (more Monte Carlo-like, less bootstrapping) |
| **Lower `gae_lambda`** (closer to 0) | Lower (more bootstrapping, but more bias) |
| **Better critic** | Lower (V(s_{t+1}) in δ_t is more accurate, so δ_t is cleaner) |
| **`n_critic_warmup_itr`** | Ensures critic is reasonable before actor trains, breaking vicious cycle |
| **`reward_scale_running`** | Normalizes reward scale → advantage scale is controlled, preventing gradient explosion |
| **`norm_adv: true`** | Normalizes advantages within batch to zero mean, unit variance → consistent gradient scale regardless of raw advantage magnitude |

---

## Part 9: Training Metrics Explained

### Ratio

**What it is**: The importance sampling ratio between the current policy and the policy that collected the data.

**Computation** (from `diffusion_ppo.py`):

```python
logratio = newlogprobs - oldlogprobs
ratio = logratio.exp()
# ...
return ratio.mean().item()  # Logged as "ratio"
```

Mathematically:

$$\text{ratio} = \frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} = \exp(\log \pi_\theta(a|s) - \log \pi_{\theta_\text{old}}(a|s))$$

**Interpretation**:

| Ratio Value | Meaning |
|-------------|---------|
| **≈ 1.0** | Current policy is similar to old policy (stable) |
| **> 1.0** | Current policy is MORE likely to take this action than old policy |
| **< 1.0** | Current policy is LESS likely to take this action than old policy |
| **>> 1.0 or << 1.0** | Policies have diverged significantly (potentially unstable) |

**What to watch for**:
- Ratio should stay close to 1.0 during training
- Large deviations (e.g., ratio > 2 or < 0.5) indicate the policy is changing too fast
- PPO clips the ratio to `[1-ε, 1+ε]` (where ε = `clip_ploss_coef`) to prevent destructive updates
- Healthy range: **0.9 - 1.1** is typical for well-tuned DPPO

**What it means if ratio drifts far from 1.0**:
- Learning rate may be too high
- `clip_ploss_coef` may be too large
- The policy may be diverging → check for training instability

### Explained Variance

**What it is**: A measure of how well the value function (critic) predicts actual returns. It is the R² score of the critic's predictions.

**Computation** (from `train_ppo_diffusion_agent.py`):

```python
y_pred = values_k.cpu().numpy()   # Critic's predictions: V(s)
y_true = returns_k.cpu().numpy()  # Actual (GAE) returns: R = A + V

var_y = np.var(y_true)
explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
```

Mathematically:

$$\text{Explained Variance} = 1 - \frac{\text{Var}(R - V(s))}{\text{Var}(R)}$$

Note: `values_k` and `returns_k` are computed at the same time, right after rollout, before any PPO updates. So this metric tells you how well the critic predicted returns **before** the current round of updates.

**Interpretation**:

| Explained Variance | Meaning |
|--------------------|---------|
| **1.0** | Perfect: critic exactly predicts returns |
| **0.0** | Critic is no better than predicting the mean |
| **< 0.0** | Critic is WORSE than predicting the mean (very bad) |
| **0.5 - 0.9** | Reasonable: critic captures most of the variance |

**Why it matters for PPO**:

The advantage is computed via GAE, which relies on the value function:

$$\hat{A}_t = \delta_t + \gamma\lambda\delta_{t+1} + ... \quad \text{where} \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

If the critic is accurate (high explained variance):
- The advantages (δ terms) are accurate signals about which actions were good/bad
- Policy updates are low-variance → stable, efficient learning

If the critic is poor (low explained variance):
- Advantages are noisy → the policy gets random/contradictory gradient signals
- Training is unstable and slow

**Typical progression**:
```
Early training:  0.0 - 0.3  (critic is learning the value landscape)
Mid training:    0.3 - 0.7  (critic is improving)
Late training:   0.7 - 0.95 (critic is well-calibrated)
```

### Other Common Metrics

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| **pg_loss** | Policy gradient loss (negative means policy improving) | Should decrease, but noisy |
| **v_loss** | Value function MSE loss: 0.5 * mean((V(s) - R)²) | Decreasing trend |
| **approx_kl** | KL divergence between old/new policy | < 0.02 typically |
| **clipfrac** | Fraction of samples where ratio was clipped | 0.05 - 0.3 |
| **eta** | DDIM stochasticity parameter (if using DDIM) | Depends on config |

---

## Part 10: Why Decreasing `act_steps` Can Increase Explained Variance

### The Observation

When you decrease `act_steps`, the explained variance tends to be higher. **This makes sense**, but the reason is more subtle than "the critic predicts over a shorter horizon."

### Clarification: The Critic Predicts the Same Thing in Both Cases

You're right that the critic V(s) predicts the **total discounted return-to-go from state s onward** regardless of `act_steps`. The prediction target is the same conceptual quantity: how much total reward will I get from here until the episode ends. Changing `act_steps` doesn't change this.

The explained variance compares V(s) to R, where R is the GAE-estimated return:

$$R_t = V(s_t) + \hat{A}_t$$

$$\hat{A}_t = \sum_{l=0}^{n-1} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\text{Explained Variance} = 1 - \frac{\text{Var}(R - V(s))}{\text{Var}(R)} = 1 - \frac{\text{Var}(\hat{A})}{\text{Var}(R)}$$

So explained variance is high when Var(A) << Var(R), i.e., when the advantage estimates have low variance relative to the returns.

### What Actually Changes With `act_steps`

The key is understanding the MDP structure that the PPO algorithm sees. Each "step" in the PPO trajectory is **one action chunk**, not one environment step. The `multi_step.py` wrapper **sums** all individual env step rewards within a chunk into a single scalar:

```python
# From multi_step.py line 170:
reward = aggregate(self.reward, "sum")  # Sum of act_steps individual env rewards
```

So each δ_t in the GAE uses this **aggregated chunk reward** r_t. Here's what changes:

**With large `act_steps` (e.g., 20):**

1. **Higher per-chunk reward variance**: The chunk reward r_t is the sum of 20 individual env step rewards. A lot can happen in 20 steps -- a grasp might succeed or fail, a box might be dropped or not, collisions might occur. The same starting state s_t can lead to very different chunk rewards r_t depending on the specific actions taken and stochastic dynamics. This means r_t has high **irreducible variance** conditioned on s_t.

2. **Noisier TD residuals**: Each δ_t = r_t + γV(s_{t+1}) - V(s_t) is noisier because r_t is noisier. Even if V is perfectly learned, the high variance in r_t makes δ_t noisy.

3. **Noisier advantage estimates**: Since A = Σ(γλ)^l δ_{t+l}, noisy δ values compound into noisy advantages. High Var(A) → low explained variance.

4. **Larger state transitions**: s_{t+1} is the state after 20 env steps, which can be very different from s_t. This makes V(s_{t+1}) less correlated with V(s_t), adding variance to δ_t even if both value estimates are good. The TD error δ has to capture a larger "gap" between what was predicted and what happened.

5. **Gamma applied per chunk, not per env step**: As noted in the code: `gamma here is applied to reward every act_steps, instead of every env step`. So γ=0.999 applied per chunk of 20 steps is effectively γ^20 ≈ 0.98 per chunk, while with act_steps=4, it's γ^4 ≈ 0.996 per chunk. Larger effective discounting per chunk means the value function changes more rapidly across chunks, making it harder to fit.

**With small `act_steps` (e.g., 4):**

1. **Lower per-chunk reward variance**: Only 4 env steps are aggregated. Less opportunity for dramatic events. The chunk reward r_t is more predictable from s_t.

2. **Cleaner TD residuals**: δ_t is less noisy because r_t has lower variance and V(s_{t+1}) is more correlated with V(s_t) (smaller state transition).

3. **Lower advantage variance**: Cleaner δ values → cleaner advantages → Var(A) is smaller → higher explained variance.

4. **Smoother value function landscape**: Consecutive states s_t, s_{t+1} are more similar. V(s) is a smoother function of state, which is easier for a neural network to approximate.

### The Core Insight

The critic predicts total return-to-go in both cases, yes. But the **GAE-computed target** R that we compare V(s) against is a **noisier estimate** of the true return when `act_steps` is large. Here's why:

The GAE return R_t = V(s_t) + A_t decomposes into the critic's own prediction plus a correction. The correction A_t depends on the TD residuals δ, which depend on the chunk rewards r_t. When chunk rewards are noisy (large act_steps), the correction A_t is noisy, so R_t bounces around a lot relative to V(s_t), even if V(s_t) is a good predictor of the true expected return.

Think of it this way:
- **True expected return** from state s: E[G | s] -- this is what V(s) is trying to learn
- **GAE return** R_t: a single-sample estimate of this, based on the actual rewards observed in one rollout

With large act_steps, the actual rewards observed deviate more from their expectation (higher variance per chunk), so R_t is a noisier estimate of E[G|s], and even a perfect V(s) can't match these noisy R_t values exactly. The residual R_t - V(s_t) = A_t has variance due to reward noise, not due to the critic being wrong.

### Analogy

Imagine predicting the average score of a basketball team. The "critic" predicts the season average (= expected total return). Now we check how well it matches:
- **Small act_steps**: Compare to scores of individual quarters. Each quarter's score has modest variance. The predicted average is close to any given quarter → high R².
- **Large act_steps**: Compare to scores of entire games. A single game's score can swing wildly due to hot/cold streaks. Even if the predicted average is correct, individual game scores deviate more → lower R².

The prediction target is the same (season average / total return), but the **individual samples** we compare against have different variance.

### Should You Just Use Small `act_steps`?

Not necessarily. Higher explained variance doesn't mean better overall training:

| Factor | Small `act_steps` | Large `act_steps` |
|--------|-------------------|-------------------|
| **Explained variance** | Higher (less reward noise per chunk) | Lower (more reward noise per chunk) |
| **Inference overhead** | More diffusion forward passes per episode | Fewer forward passes |
| **Action consistency** | May be jerky (replans frequently) | Smoother (commits to longer plans) |
| **Credit assignment** | Reward signal is fine-grained | Reward is aggregated, harder to assign credit |
| **Exploration** | More reactive to new observations | Commits to action sequences |
| **Boundary effects** | More chunk transitions (potential discontinuities) | Fewer transitions |

The optimal `act_steps` is task-dependent. For tasks requiring reactive, precise manipulation (like truck unloading), smaller `act_steps` can give cleaner training signals. For tasks where committing to a plan matters, larger `act_steps` may be preferable despite lower explained variance.

---

## Summary

1. **Variance is fixed**: The log variance σ² comes from the diffusion schedule (DDPM) or eta model (DDIM), NOT from the neural network. The network only affects the mean μ.

2. **DDPM vs DDIM mean**: μ is computed differently - DDPM uses a weighted sum of x₀ and x_t, while DDIM uses a weighted sum of x₀ and ε. Both depend on the network's noise prediction.

3. **oldlogprobs vs newlogprobs**: oldlogprobs are computed once after rollout with the rollout weights. newlogprobs are recomputed during PPO updates with the current (updated) weights. They match initially but diverge as the policy is updated.

4. **Gradient flow**: pg_loss → ratio → newlogprobs → log_prob → μ → x_recon → noise_pred → actor_ft weights. Everything else (chains, oldlogprobs, σ, advantages) is detached.

5. **`reward_horizon`** (defaults to `act_steps`): Only backpropagates through the action steps that were actually executed in the environment. Prevents noisy gradients from unexecuted actions.

6. **`min_logprob_denoising_std`**: Prevents gradient explosion by clamping σ² in the gradient formula ∂log_prob/∂μ = (x-μ)/σ².

7. **`min_sampling_denoising_std`**: Controls exploration noise during rollout.

8. **Ratio ≈ 1.0** indicates stable training; **Explained Variance > 0.5** indicates a well-calibrated critic.

9. **GAE** estimates A = Q - V by interpolating between low-variance/high-bias (TD) and high-variance/zero-bias (Monte Carlo) estimators. Lower variance advantages lead to more stable training, better sample efficiency, and a virtuous cycle with critic learning.

10. **Smaller `act_steps` → higher explained variance**: Not because the critic predicts over a shorter horizon (it always predicts total return-to-go), but because the per-chunk reward has lower variance, making the GAE return targets less noisy. Even a perfect critic can't explain variance that comes from reward stochasticity within chunks.
