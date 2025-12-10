# Truck Unloading Task: DPPO Baseline for MPI Comparison

## Overview

This document describes the integration of Diffusion Policy Policy Optimization (DPPO) as a baseline method for comparing against our Motion Policy Improvement (MPI) expert iteration approach. The goal is to finetune the same pretrained policy and value network checkpoints using DPPO and compare the results with MPI's iterative refinement.

## Motivation

Our MPI method trains diffusion policies through an iterative process combining:

- Behavioral cloning on expert demonstrations
- Value network training for action evaluation
- Expert iteration with beam search for trajectory improvement

DPPO provides an alternative approach using reinforcement learning (PPO) to directly finetune the diffusion policy. By starting from identical pretrained checkpoints, we can fairly compare:

- Sample efficiency
- Final task performance
- Training stability

## Architecture Differences

### MPI Models (ResNet-based)

Our MPI repository uses custom architectures:

1. **Policy Network**: `DiffusionUnetHybridImageTargetedPolicy`

   - Image encoder: RoboMimic-based CNN (not ViT)
   - Denoiser: `ConditionalUnet1D` for action trajectory prediction
   - Observation history: 2 timesteps
   - Output: 32-step action trajectory (20 steps executed)
2. **Value Network**: `ResNet18TVTruck2dValueNetwork`

   - Backbone: TorchVision ResNet18 (Not pretrained)
   - Observation: Single timestep (current state only)
   - Output: Scalar value estimate

### DPPO Default Models (ViT-based)

The original DPPO implementation uses:

- Vision Transformer (ViT) for image encoding
- Different observation/action dimensions
- Different normalization conventions

## Integration Approach

Rather than modifying our model architectures to match DPPO's expectations, we created wrapper layers that adapt MPI models to DPPO's training interface while preserving the original architectures.

### Key Design Decisions

1. **Clean Hydra Integration**: The model is instantiated directly via Hydra config targeting `MPIPPODiffusion.from_mpi_checkpoints()`, avoiding any placeholder model loading/replacement.
2. **Preserve Original Architectures**: MPI policy and value networks are loaded exactly as trained, with wrappers handling interface translation.
3. **Use MPI's Noise Scheduler**: For inference, we use MPI's diffusers-based `DDPMScheduler` with `squaredcos_cap_v2` beta schedule to ensure identical sampling behavior.
4. **Consistent Normalization**: All normalization (images, state, actions) uses MPI's `LinearNormalizer` to match training conventions.

## Implementation Details

### File Structure

```
dppo-baseline/
├── model/
│   ├── diffusion/
│   │   └── mpi_ppo_diffusion.py      # Main integration: MPIPPODiffusion, MPIPolicyActorWrapper
│   └── common/
│       └── mpi_critic_wrapper.py      # MPICriticWrapper for value network
├── agent/finetune/
│   └── train_ppo_diffusion_truck2d_agent.py  # Training agent with custom metrics
├── cfg/truck_2d/
│   ├── ft_ppo_mpi_truck_2d.yaml       # Full training config
│   └── ft_ppo_mpi_truck_2d_test.yaml  # Test/debug config
└── env/gym_utils/wrapper/
    └── multi_step.py                   # Multi-step action wrapper
```

### Core Components

#### 1. MPIPolicyActorWrapper (`mpi_ppo_diffusion.py`)

Wraps MPI's policy components for DPPO's actor interface:

```python
class MPIPolicyActorWrapper(nn.Module):
    def __init__(self, obs_encoder, unet_model, normalizer, ...):
        self.obs_encoder = obs_encoder  # MPI's image encoder
        self.model = unet_model         # ConditionalUnet1D
        self.normalizer = normalizer    # LinearNormalizer
  
    def _encode_obs(self, cond):
        # Transform DPPO's cond dict to MPI's expected format
        # Normalize images: [0,1] -> [-1,1]
        # Normalize state using 'robot_state' key
        # Encode with obs_encoder -> global_cond vector
  
    def forward(self, x, t, cond):
        # DPPO calls: model(x, t, cond)
        # MPI expects: model(sample, timestep, global_cond)
        global_cond = self._encode_obs(cond)
        return self.model(sample=x, timestep=t, global_cond=global_cond)
```

#### 2. MPICriticWrapper (`mpi_critic_wrapper.py`)

Adapts MPI's value network for DPPO's critic interface:

```python
class MPICriticWrapper(nn.Module):
    def forward(self, cond):
        # Extract latest observation (critic uses 1 timestep, not 2)
        # Reshape to MPI value network's expected format
        # Call value_network.predict_value() which handles internal normalization
```

#### 3. MPIPPODiffusion (`mpi_ppo_diffusion.py`)

Main diffusion model class combining everything:

```python
class MPIPPODiffusion(nn.Module):
    @classmethod
    def from_mpi_checkpoints(cls, policy_checkpoint_path, critic_checkpoint_path, ...):
        # Load MPI policy with DDP prefix handling
        # Load MPI value network
        # Set normalizers
        # Return configured MPIPPODiffusion instance
  
    def forward(self, cond, deterministic=False, return_chain=True):
        # Sample actions using MPI's DDPMScheduler
        # Returns Sample(trajectories, chains) for DPPO training
  
    def get_logprobs(self, cond, chains, ...):
        # Compute log probabilities for PPO loss
        # Uses synced alphas/betas from MPI scheduler
```

### Normalization Pipeline

Critical for correct behavior - MPI's normalizer transforms:

| Data        | Raw Range    | Normalized Range | Normalizer Key  |
| ----------- | ------------ | ---------------- | --------------- |
| Images      | [0, 1]       | [-1, 1]          | `images`      |
| Robot State | varies       | [-1, 1]          | `robot_state` |
| Actions     | joint limits | [-1, 1]          | `action`      |

**Flow during inference:**

1. Environment returns images in [0, 1], state in raw joint values
2. `MPIPolicyActorWrapper._encode_obs()` normalizes both using MPI normalizer
3. Policy outputs normalized actions in [-1, 1]
4. Agent's `_unnormalize_actions()` converts to raw joint positions
5. Environment receives raw joint positions (`normalize_actions: false`)

### Noise Scheduler Integration

To ensure identical sampling behavior with MPI:

```python
# Store MPI's scheduler
self.mpi_noise_scheduler = mpi_policy.noise_scheduler  # DDPMScheduler

# Sync DDPM parameters for log prob computation
self.alphas_cumprod = scheduler.alphas_cumprod  # From squaredcos_cap_v2
self.betas = scheduler.betas

# Sampling uses scheduler.step() exactly like MPI's conditional_sample
for t in scheduler.timesteps:
    noise_pred = actor.model(x, t, global_cond)
    x = scheduler.step(noise_pred, t, x).prev_sample
```

## Configuration

### Key Config Parameters (`ft_ppo_mpi_truck_2d_test.yaml`)

```yaml
# MPI Checkpoint Paths
base_policy_path: /path/to/policy/checkpoint.ckpt
critic_path: /path/to/value_network/checkpoint.pt
policy_normalizer_path: /path/to/normalizer.pt
critic_normalizer_path: /path/to/value_network_normalizer.pt

# Diffusion settings (match MPI training)
denoising_steps: 100      # Total DDPM steps
ft_denoising_steps: 10    # Finetune last N steps only
horizon_steps: 32         # Action trajectory length
act_steps: 16             # Steps actually executed

# Observation dimensions
cond_steps: 2             # Policy observation history
critic_n_obs_steps: 1     # Critic observation history (current only)

# Environment
env:
  normalize_actions: false  # Env expects raw joint positions
```

### Critic Options: Pretrained vs Fresh

The system supports two modes for the value network (critic):

#### Option 1: Pretrained Critic (Default)

Load a pretrained MPI value network checkpoint. The normalizer is frozen and doesn't update during training.

```yaml
model:
  _target_: model.diffusion.mpi_ppo_diffusion.MPIPPODiffusion.from_mpi_checkpoints
  policy_checkpoint_path: /path/to/policy.ckpt
  policy_normalizer_path: /path/to/normalizer.pt
  critic_checkpoint_path: /path/to/critic.pt
  critic_normalizer_path: /path/to/critic_normalizer.pt
  use_pretrained_critic: true  # default
```

**Pros:**
- Critic starts with reasonable value estimates
- Faster initial training

**Cons:**
- Normalizer is frozen (won't adapt to distribution shift during PPO)
- Requires a pretrained value network checkpoint

#### Option 2: Fresh Critic (Trained from Scratch)

Train the critic from random initialization during PPO fine-tuning. Uses the policy normalizer for observation normalization.

```yaml
model:
  _target_: model.diffusion.mpi_ppo_diffusion.MPIPPODiffusion.from_mpi_checkpoints
  policy_checkpoint_path: /path/to/policy.ckpt
  policy_normalizer_path: /path/to/normalizer.pt
  use_pretrained_critic: false
  critic:
    _target_: value_network.models.resnet18_tv_truck_2d_value_network.ResNet18TVTruck2dValueNetwork
    img_shape: [1, 71, 192]
    state_dim: 6
    features_dim: 512
```

**Pros:**
- Critic learns alongside the policy (no distribution mismatch)
- No need for pretrained value network checkpoint
- Standard approach in PPO implementations

**Cons:**
- Requires warmup iterations (`n_critic_warmup_itr`) for critic to learn initial estimates
- May need higher critic learning rate initially

**Training Tips for Fresh Critic:**
- Set `n_critic_warmup_itr: 5` or higher to let critic learn before actor updates
- Use higher `critic_lr` (e.g., 1e-3) for faster initial learning
- Consider more `update_epochs` per iteration

See `cfg/truck_2d/ft_ppo_mpi_truck_2d_fresh_critic.yaml` for a complete example.

## Usage

```bash
# From motion-policy-improvement directory (for environment access)
cd /path/to/motion-policy-improvement

# Run training
poetry run python ../dppo-baseline/script/run.py \
    --config-path=/path/to/dppo-baseline/cfg/truck_2d \
    --config-name=ft_ppo_mpi_truck_2d_test
```

## Custom Metrics

The training agent tracks truck unloading-specific metrics:

- **avg_pieces_per_hour**: Average throughput (boxes removed per hour)
- **avg_task_completion**: Ratio of boxes removed to total boxes
- **custom_success_rate**: Based on whether all boxes were removed
- **n_episodes_completed**: Number of completed episodes per iteration

## Challenges and Solutions

### 1. Checkpoint Loading with OmegaConf

**Problem**: MPI checkpoints contain OmegaConf objects, failing `torch.load(..., weights_only=True)`.
**Solution**: Use `weights_only=False` in `from_mpi_checkpoints()` and handle DDP `module.` prefix stripping.

### 2. Observation History Mismatch

**Problem**: Policy uses 2 timesteps, critic uses 1.
**Solution**: `MPICriticWrapper` extracts only the latest observation for value estimation.

## References

- [DPPO Paper](https://arxiv.org/abs/2409.00588) - Diffusion Policy Policy Optimization
- MPI Repository: `motion-policy-improvement/`
- Environment: `motion-policy-improvement/dppo_baseline/env/truck_2d_env.py`
