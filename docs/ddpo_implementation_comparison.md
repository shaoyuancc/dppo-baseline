# DPPO Implementation Comparison: Default Robomimic vs MPI Truck Unloading

This document systematically compares the default DPPO implementation (for robomimic tasks like Square) with the MPI truck unloading DPPO implementation. The goal is to identify differences that might explain why DPPO works for simpler tasks (reach, single box) but not for the 4-box truck unloading task.

## Table of Contents

1. [Environment Differences](#1-environment-differences)
2. [Reward Structure](#2-reward-structure)
3. [Action Space and Trajectory Handling](#3-action-space-and-trajectory-handling)
4. [Policy Architecture](#4-policy-architecture)
5. [Critic Architecture](#5-critic-architecture)
6. [Observation Space](#6-observation-space)
7. [Diffusion Schedule and Sampling](#7-diffusion-schedule-and-sampling)
8. [Normalization](#8-normalization)
9. [Training Configuration](#9-training-configuration)
10. [Agent Implementation](#10-agent-implementation)
11. [Potential Experiments](#11-potential-experiments)

---

## 1. Environment Differences

### 1.1 Episode Termination

| Aspect | Robomimic (Default DPPO) | MPI Truck 2D |
|--------|--------------------------|--------------|
| **Episode termination** | Fixed length, never terminates early even on success | Variable length, early terminates on success or failure |
| **Success handling** | Continues running; agent receives reward 1 for all remaining steps after success | Episode ends immediately on success with success_reward |
| **Failure handling** | Episode continues (no explicit failure in these tasks) | Episode ends on penetration failure, tracking failure, or timeout |
| **Max episode steps** | 300-800 (task-dependent, see table below) | Variable: 50 (reach), 200 (single box), 800 (4 box) |
| **Steps per policy query** | 4-8 environment steps (task-dependent) | 20 environment steps |

**Robomimic Task Parameters:**

| Task | obs_dim | action_dim | horizon_steps | act_steps | max_episode_steps | n_train_itr | actor_lr |
|------|---------|------------|---------------|-----------|-------------------|-------------|----------|
| Lift | 19 | 7 | 4 | 4 | 300 | 81 | 1e-4 |
| Can | 23 | 7 | 4 | 4 | 300 | 151 | 1e-4 |
| Square | 23 | 7 | 4 | 4 | 400 | 201 | 1e-5 |
| Transport | 59 | 14 | 16 | 8 | 800 | 201 | 2e-5 |

**Implications:**
- Robomimic's fixed-length episodes with continued reward after success naturally incentivize completing tasks quickly (more reward steps after success)
- MPI's early termination requires explicit time-cost penalties in the reward to incentivize speed
- MPI's longer action horizon (20 steps vs 4-8) means each policy decision covers much more time
- Transport task has longer horizons (16/8) and larger action space (14D for two arms)

### 1.2 Environment Type

| Aspect | Robomimic | MPI Truck 2D |
|--------|-----------|--------------|
| **Simulation** | MuJoCo via robomimic | Drake physics |
| **Control mode** | End-effector delta/absolute pose | Absolute joint angles |
| **Task complexity** | Single manipulation goal | Multi-stage: approach → grasp → return → drop (per box) |
| **Observation** | Low-dim state only (19-59D) | 6D robot state + (1, 71, 192) depth image |

### 1.3 Environment Wrapper

| Aspect | Robomimic | MPI Truck 2D |
|--------|-----------|--------------|
| **Wrapper** | `MultiStep` | `MultiStep` |
| **Trajectory execution** | Steps through actions one at a time | Steps through actions one at a time |
| **Reset behavior** | `reset_within_step: True` | `reset_within_step: True` |

Note: `FullTrajectoryMultiStep` wrapper exists but is NOT currently used in truck 2d configs. All configs use the standard `MultiStep` wrapper.

---

## 2. Reward Structure

### 2.1 Robomimic Sparse Reward

```yaml
# Robomimic Square task
reward: 0 or 1 (sparse binary)
best_reward_threshold_for_success: 1
```

- **During episode:** 0 until success, then 1 for each remaining step
- **Episode sum:** Number of steps after success
- **Incentive:** Complete task quickly to maximize cumulative reward

### 2.2 MPI Dense Reward Configuration

From `ft_ppo_mpi_truck_2d_single_box_fresh_critic.yaml`:

```yaml
reward_config:
  # Base rewards
  step_reward: -0.1          # Per-step time penalty (always negative)
  failure_reward: -20.0      # Penetration/tracking failure
  success_reward: 10.0       # All boxes removed
  
  # Dense reward mode
  dense_reward_enabled: true
  
  # Event rewards
  grasp_reward: 1.0          # Successful grasp (suction on → attached)
  failed_grasp_penalty: -1.0 # Failed grasp attempt
  drop_reward: 2.0           # Successful drop in drop zone
  failed_drop_penalty: -2.0  # Released box outside drop zone
  
  # Shaping
  max_reward_distance: 3.0   # Distance beyond which shaping is 0
  dense_reward_scale: 0.05   # Scale for distance-based shaping
  
  # Grasp maintenance
  maintain_grasp_reward: 0.02 # Binary: correct suction behavior
```

**Dense Reward Computation (from `truck_2d_env.py`):**

```python
def _compute_dense_reward(self, ...):
    reward = step_reward  # Start with -0.1
    
    # Stage-dependent shaping
    if is_suctioned:
        # RETURNING stage: reward getting closer to drop zone
        dist = self._compute_distance_to_drop_zone()
        # Add maintain_grasp_reward for correct suction behavior
        reward += maintain_grasp_reward if correct_suction else -maintain_grasp_reward
    else:
        # APPROACHING stage: reward getting closer to nearest box
        dist = self._compute_min_distance_to_box()
    
    # Distance shaping: dense_reward_scale * max(0, 1 - dist/max_dist)
    shaping_reward = dense_reward_scale * max(0, 1 - dist/max_reward_distance)
    reward += shaping_reward
    
    # Event rewards for grasp/drop transitions
    if grasp_attempt:
        reward += grasp_reward if successful else failed_grasp_penalty
    if release_event:
        reward += drop_reward if successful else failed_drop_penalty
```

### 2.3 Reach Task Reward (Simplified)

```yaml
reward_config:
  step_reward: -0.1
  failure_reward: -10.0
  success_reward: 10.0
  touch_threshold: 0.05
  max_reward_distance: 3.0
  dense_reward_scale: 0.08
```

**Key Differences:**
| Aspect | Robomimic | MPI (Dense) |
|--------|-----------|-------------|
| Sparsity | Binary 0/1 | Dense multi-component |
| Time incentive | Implicit (more 1s after success) | Explicit step penalty (-0.1) |
| Stage awareness | None | Stage-dependent shaping |
| Event rewards | None | Grasp/drop events |
| Distance shaping | None | Continuous distance-based |

---

## 3. Action Space and Trajectory Handling

### 3.1 Action Space Comparison

| Aspect | Robomimic | MPI Truck 2D |
|--------|-----------|--------------|
| **Action dimension** | 7 (single arm) or 14 (Transport, two arms) | 5 (4 joints + suction) |
| **Action type** | End-effector delta or absolute (6 DoF + gripper) | Absolute joint angles |
| **Prediction horizon** | 4 (simple) or 16 (Transport) | 32 steps |
| **Action horizon** | 4 (simple) or 8 (Transport) | 20 steps |
| **Effective horizon ratio** | 1:1 | 32:20 (predicts more than executes) |

### 3.2 Trajectory Execution

Both use `MultiStep` wrapper which steps through actions one at a time:

```python
# MultiStep wrapper (used by both Robomimic and MPI Truck 2D)
for act in action[:n_action_steps]:  # Iterate through act_steps
    obs, reward, done, info = env.step(act)
```

Note: `FullTrajectoryMultiStep` wrapper exists for MPI (creates smooth `PiecewisePolynomial` trajectories) but is **NOT currently used** in the truck 2d configs.

### 3.3 Impact on Effective Episode Horizon

| Task | Max Env Steps | Act Steps | Horizon Steps | Max Policy Steps |
|------|---------------|-----------|---------------|------------------|
| Robomimic Lift | 300 | 4 | 4 | 75 |
| Robomimic Can | 300 | 4 | 4 | 75 |
| Robomimic Square | 400 | 4 | 4 | 100 |
| Robomimic Transport | 800 | 8 | 16 | 100 |
| MPI Reach | 50 | 20 | 32 | 2.5 |
| MPI Single Box | 200 | 20 | 32 | 10 |
| MPI 4 Box | 800 | 20 | 32 | 40 |

**Key Observations:** 
- MPI tasks have much fewer effective policy decisions despite similar or longer episode lengths
- MPI predicts 32 steps but only executes 20, while Robomimic (except Transport) predicts and executes the same number
- Transport is the closest Robomimic task to MPI in terms of horizon ratio (16:8 vs 32:20)

---

## 4. Policy Architecture

### 4.1 Robomimic DPPO Policy (`Unet1D`)

```yaml
model:
  actor:
    _target_: model.diffusion.unet.Unet1D
    diffusion_step_embed_dim: 16
    dim: 40-64  # 40 for lift/can, 64 for square/transport
    dim_mults: [1, 2]  # 2 levels
    kernel_size: 5
    n_groups: 8
    smaller_encoder: False
    cond_predict_scale: True
    cond_dim: obs_dim * cond_steps  # varies by task
    action_dim: 7 or 14  # 7 single-arm, 14 transport
```

**Task-specific UNet settings:**
| Task | dim | cond_dim (obs_dim × cond_steps) | action_dim |
|------|-----|----------------------------------|------------|
| Lift | 40 | 19 × 1 = 19 | 7 |
| Can | 40 | 23 × 1 = 23 | 7 |
| Square | 64 | 23 × 1 = 23 | 7 |
| Transport | 64 | 59 × 1 = 59 | 14 |

- **Observation:** State-only (`cond_steps: 1`)
- **Image encoder:** None for low-dim tasks
- **UNet:** Small (2 level, dim 40-64)

### 4.2 Robomimic DPPO Image Policy (`VisionUnet1D`)

```yaml
model:
  actor:
    _target_: model.diffusion.unet.VisionUnet1D
    backbone:
      _target_: model.common.vit.VitEncoder
      cfg:
        patch_size: 8
        depth: 1
        embed_dim: 128
        num_heads: 4
    img_cond_steps: 1
    spatial_emb: 128
    diffusion_step_embed_dim: 32
    dim: 40-64  # 40 for lift, 64 for square/transport
    dim_mults: [1, 2]
    cond_dim: obs_dim * cond_steps  # 9 or 18 (robot state only)
```

- **Observation:** Robot state only (9D or 18D) + image (`cond_steps: 1`, `img_cond_steps: 1`)
- **Image encoder:** ViT (patch_size=8, depth=1, embed_dim=128)
- **Object state:** NOT included in state - must be inferred from image
- **UNet:** Same as low-dim (2 level, dim 40-64)
- **Diffusion:** Uses DDIM with 100 steps, 5 ft steps

### 4.3 MPI Policy (`DiffusionUnetHybridImageTargetedPolicy`)

```python
class DiffusionUnetHybridImageTargetedPolicy:
    # Image encoder: RoboMimic CNN (not ViT)
    obs_encoder = robomimic.policy.encoder  # ResNet-based
    
    # Denoiser: ConditionalUnet1D
    model = ConditionalUnet1D(
        input_dim=action_dim,  # 5
        global_cond_dim=obs_feature_dim * n_obs_steps,  # ~512 * 2
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),  # 3 levels
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
    )
```

- **Observation:** 2 history steps (`n_obs_steps: 2`)
- **Image encoder:** RoboMimic CNN (from `bc_rnn` algo)
- **UNet:** Larger (3 levels, dims 256-512-1024)

### 4.4 Architecture Comparison

| Aspect | Robomimic Low-Dim | Robomimic Image | MPI |
|--------|-------------------|-----------------|-----|
| **Image encoder** | None | ViT (shallow) | RoboMimic CNN |
| **UNet levels** | 2 (dim_mults: [1,2]) | 2 (dim_mults: [1,2]) | 3 (down_dims: 256,512,1024) |
| **Base dim** | 40-64 | 40-64 | 256 |
| **Diffusion embed dim** | 16 | 32 | 256 |
| **Cond predict scale** | True | True | True |
| **State observation** | Full (robot + object) | Robot only | Robot only (6D) |
| **Object info source** | State vector | Image | Depth image |
| **Observation history** | 1 step | 1 step | 2 steps |
| **Denoising steps** | 20 | 100 | 100 |
| **FT denoising steps** | 10 | 5 (DDIM) | 5-10 (DDIM) |

---

## 5. Critic Architecture

### 5.1 Robomimic DPPO Low-Dim Critic (`CriticObs`)

```yaml
critic:
  _target_: model.common.critic.CriticObs
  mlp_dims: [256, 256, 256]
  activation_type: Mish
  residual_style: True
  cond_dim: 19-59  # obs_dim * cond_steps (includes object state)
```

- **Input:** Same as policy (full state including object, 1 step)
- **Architecture:** 3-layer MLP with residual connections
- **Parameters:** ~200K

### 5.2 Robomimic DPPO Image Critic (`ViTCritic`)

```yaml
critic:
  _target_: model.common.critic.ViTCritic
  spatial_emb: 128
  backbone:
    _target_: model.common.vit.VitEncoder
    cfg:
      patch_size: 8
      depth: 1
      embed_dim: 128
      num_heads: 4
  img_cond_steps: 1
  mlp_dims: [256, 256, 256]
  cond_dim: 9-18  # robot state only (no object state)
```

- **Input:** Robot state (no object) + image
- **Architecture:** ViT backbone + MLP head (same as actor backbone)
- **Object info:** Inferred from image (no privileged info)

### 5.3 MPI Truck 2D Critic (`ResNet18TVTruck2dValueNetwork`)

```python
class ResNet18TVTruck2dValueNetwork:
    # Vision backbone
    backbone = torchvision.resnet18(weights=None)  # Modified for 1-channel
    global_pool = AdaptiveAvgPool2d(1)  # -> 512 features
    
    # Vector branch  
    vector_branch = Sequential(
        Linear(state_dim, 128), ReLU,
        Linear(128, 64), ReLU
    )
    
    # Value head
    value_head = Sequential(
        Linear(512 + 64, 256), ReLU,
        Linear(256, 128), ReLU,
        Linear(128, 1),
        # Tanh() removed for DPPO (use_tanh=False)
    )
```

- **Input:** Latest single frame only (`critic_n_obs_steps: 1`)
- **Architecture:** ResNet-18 backbone + MLP head
- **Parameters:** ~11M (much larger)

### 5.4 Asymmetric Actor-Critic Option (MPI)

MPI supports asymmetric critic with `full_state` privileged information:

```python
# full_state = [station_state, suction_info, box_geometries, min_distance_to_box]
# Dimension: 11 + max_boxes * 8 (e.g., 43 for 4 boxes)
full_state_dim = 11 + max_boxes * 8
```

### 5.5 Key Critic Differences

| Aspect | Robomimic Low-Dim | Robomimic Image | MPI |
|--------|-------------------|-----------------|-----|
| **Architecture** | MLP (256×3) | ViT + MLP | ResNet-18 + MLP |
| **Input** | Same as policy | Same as policy | Different (single frame) |
| **Parameters** | ~200K | ~500K | ~11M |
| **Image processing** | None | ViT (shallow) | Full ResNet-18 |
| **Object state** | In state vector | From image | From depth + `full_state` option |
| **Privileged info** | None (but has object) | None | `full_state` option |
| **Output activation** | Linear | Linear | Linear (`use_tanh: False`) |

---

## 6. Observation Space

### 6.1 Robomimic Low-Dim Observation

```yaml
# From robomimic_lowdim wrapper - varies by task
# Square/Can:
low_dim_keys: 
  - robot0_eef_pos       # 3D
  - robot0_eef_quat      # 4D
  - robot0_gripper_qpos  # 2D
  - object               # 14D (Square) or 14D (Can)
# Total: 23D state

# Lift:
low_dim_keys: 
  - robot0_eef_pos       # 3D
  - robot0_eef_quat      # 4D
  - robot0_gripper_qpos  # 2D
  - object               # 10D
# Total: 19D state

# Transport (two-arm):
low_dim_keys: 
  - robot0_eef_pos       # 3D
  - robot0_eef_quat      # 4D
  - robot0_gripper_qpos  # 2D
  - robot1_eef_pos       # 3D
  - robot1_eef_quat      # 4D
  - robot1_gripper_qpos  # 2D
  - object               # 41D
# Total: 59D state
```

Normalization: Min-max to [-1, 1] using precomputed statistics.

### 6.2 Robomimic Image Observation

**Key difference: Image tasks do NOT include object state - only robot proprioception.**

```yaml
# From robomimic_image wrapper - object state excluded
# Lift/Square/Can (img):
low_dim_keys: 
  - robot0_eef_pos       # 3D
  - robot0_eef_quat      # 4D
  - robot0_gripper_qpos  # 2D
# Total: 9D state + image
image_keys: ['agentview_image']  # or 'robot0_eye_in_hand_image'
# Image shape: [3, 96, 96]

# Transport (img, two-arm):
low_dim_keys: 
  - robot0_eef_pos       # 3D
  - robot0_eef_quat      # 4D
  - robot0_gripper_qpos  # 2D
  - robot1_eef_pos       # 3D
  - robot1_eef_quat      # 4D
  - robot1_gripper_qpos  # 2D
# Total: 18D state + images
image_keys: ['shouldercamera0_image', 'shouldercamera1_image']
# Image shape: [6, 96, 96] (2 images × 3 channels)
```

**Observation Comparison Table:**

| Task | Low-Dim obs_dim | Image obs_dim | Object state included? |
|------|-----------------|---------------|------------------------|
| Lift | 19 | 9 + image | Low-dim: Yes, Image: No |
| Can | 23 | 9 + image | Low-dim: Yes, Image: No |
| Square | 23 | 9 + image | Low-dim: Yes, Image: No |
| Transport | 59 | 18 + 2 images | Low-dim: Yes, Image: No |

**This is a key difference from MPI:** Robomimic image tasks must infer object state from images, whereas MPI truck 2D provides object information via the depth image but also has `full_state` with privileged object info for the critic.

### 6.3 MPI Truck 2D Observation

From `truck_2d_env.py`:

```python
observation_space = spaces.Dict({
    "state": Box(shape=(6,)),          # 4 joints + suction_cmd + is_suctioned
    "rgb": Box(shape=(1, 71, 192)),    # Grayscale depth image [0, 1]
    "full_state": Box(shape=(full_state_dim,)),  # Privileged state for critic
})

# Action space
action_space = Box(low=-1.0, high=1.0, shape=(5,))  # 4 joint positions + 1 suction command
```

Full state structure (from code):
```python
# full_state = [station_state, last_suction_cmd, is_suctioned, box_geometries, min_distance_to_box]
# station_state: 4 robot_pos + max_boxes*3 box_pos + 4 robot_vel + max_boxes*3 box_vel
#              = 8 + max_boxes * 6
# box_geometries: max_boxes * 2 (w, h per box)
# min_distance_to_box: 1 (signed distance from suction tip to nearest box)
# Total: (8 + max_boxes * 6) + 2 + max_boxes * 2 + 1 = 11 + max_boxes * 8

full_state_dim = 11 + max_boxes * 8
# 1 box:  11 + 1*8 = 19
# 2 boxes: 11 + 2*8 = 27  
# 4 boxes: 11 + 4*8 = 43
```

### 6.4 Observation Dimensions Summary

| Component | Robomimic | MPI Truck 2D |
|-----------|-----------|--------------|
| **State dim** | 19-59 (task-dependent) | 6 (fixed) |
| **Image** | None (low-dim only) | (1, 71, 192) depth |
| **Action dim** | 7 (single arm) or 14 (two arm) | 5 |
| **Privileged state** | None | 19-43 (box-count dependent) |

### 6.5 Observation History

| Aspect | Robomimic | MPI Policy | MPI Critic |
|--------|-----------|------------|------------|
| **State history** | 1 step | 2 steps | 1 step |
| **Image history** | N/A | 2 steps | 1 step |

---

## 7. Diffusion Schedule and Sampling

### 7.1 DDPM vs DDIM

| Parameter | Robomimic | MPI |
|-----------|-----------|-----|
| **Denoising steps** | 20 | 100 |
| **FT denoising steps** | 10 | 5-10 |
| **Use DDIM** | Optional | Yes (`use_ddim: true`) |
| **DDIM steps** | - | 5-10 |
| **Eta** | Fixed | `EtaFixed(base_eta=1.0)` |

### 7.2 Noise Schedule

| Aspect | Robomimic | MPI |
|--------|-----------|-----|
| **Beta schedule** | Cosine (custom) | `squaredcos_cap_v2` (diffusers) |
| **Scheduler** | Custom DDPM | diffusers `DDPMScheduler` |

### 7.3 Sampling Std

```yaml
# Robomimic
min_sampling_denoising_std: 0.1
min_logprob_denoising_std: 0.1

# MPI
min_sampling_denoising_std: 0.01  # Much lower
min_logprob_denoising_std: 0.01  # or 0.1 for reach
```

---

## 8. Normalization

### 8.1 Robomimic Normalization

```python
# From robomimic_lowdim.py
def normalize_obs(self, obs):
    # Min-max normalization to [-1, 1]
    obs = 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)
    return obs

def unnormalize_action(self, action):
    # [-1, 1] -> [action_min, action_max]
    action = (action + 1) / 2  # -> [0, 1]
    return action * (self.action_max - self.action_min) + self.action_min
```

- Uses precomputed statistics from `normalization.npz`
- Applied in environment wrapper

### 8.2 MPI Normalization

```python
# From LinearNormalizer (diffusion_policy)
# Images: [0, 1] -> [-1, 1] (scale=2, offset=-1)
# State: computed from dataset statistics
# Actions: computed from dataset (joint limits)
```

- Uses `LinearNormalizer` from diffusion_policy
- Applied in `MPIPolicyActorWrapper._encode_obs()` for observations
- Applied in `TrainPPODiffusionTruck2DAgent._unnormalize_actions()` for actions

### 8.3 Key Difference

| Location | Robomimic | MPI |
|----------|-----------|-----|
| **Obs normalization** | In env wrapper | In policy wrapper |
| **Action normalization** | In env wrapper | In agent |
| **Normalizer type** | Min-max | LinearNormalizer |
| **Statistics source** | Precomputed file | Learned from dataset |

---

## 9. Training Configuration

### 9.1 Hyperparameters Comparison

| Parameter | Robomimic Lift | Robomimic Square | Robomimic Transport | MPI Single Box | MPI 4 Box |
|-----------|----------------|------------------|---------------------|----------------|-----------|
| **n_train_itr** | 81 | 201 | 201 | 201 | 201 |
| **n_critic_warmup_itr** | 2 | 2 | 2 | 2 | 2 |
| **n_steps** | 300 | 400 | 400 | 180 | 180 |
| **n_envs** | 50 | 50 | 50 | 50 | 50 |
| **gamma** | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| **gae_lambda** | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| **actor_lr** | 1e-4 | 1e-5 | 2e-5 | 1e-5 | 1e-5 |
| **critic_lr** | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| **batch_size** | 7500 | 10000 | 10000 | 500 | 500 |
| **update_epochs** | 10 | 10 | 5 | 10 | 10 |
| **vf_coef** | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| **target_kl** | 1 | 1 | 1 | 1.0 | 1.0 |
| **grad_accumulate** | N/A | N/A | N/A | 20 | 20 |
| **reward_scale_running** | True | True | True | True | True |

### 9.2 PPO-Specific Parameters

```yaml
# Both use same PPO parameters
gamma_denoising: 0.99
clip_ploss_coef: 0.01
clip_ploss_coef_base: 0.001
clip_ploss_coef_rate: 3
norm_adv: true
```

### 9.3 Notable Differences

| Aspect | Robomimic | MPI |
|--------|-----------|-----|
| **Batch size** | 7500-10000 | 500 (with grad_accumulate=20) |
| **Effective batch** | 7500-10000 | 10000 |
| **Steps per iteration** | 300-400 | 180 |
| **Samples per iteration** | 300-400 × 50 = 15-20K | 180 × 50 = 9K |
| **Actor LR range** | 1e-4 to 2e-5 | 1e-5 |
| **Update epochs** | 5-10 | 10 |

---

## 10. Agent Implementation

### 10.1 Training Loop Differences

**Robomimic (`TrainPPODiffusionAgent`):**
```python
# Simple state-only processing
cond = {"state": torch.from_numpy(prev_obs_venv["state"])}
samples = self.model(cond=cond, ...)
action_venv = samples.trajectories[:, :self.act_steps]
# No action unnormalization - env wrapper handles it
obs_venv, reward, ... = self.venv.step(action_venv)
```

**MPI (`TrainPPODiffusionTruck2DAgent`):**
```python
# Multi-modal observation handling
cond = {
    key: torch.from_numpy(prev_obs_venv[key]) 
    for key in self.obs_dims  # state, rgb, full_state
}
samples = self.model(cond=cond, ...)

# Full trajectory mode support
if self.use_full_trajectory_mode:
    action_venv = output_venv  # Full horizon for smooth trajectories
else:
    action_venv = output_venv[:, :self.act_steps]

# Manual action unnormalization
action_venv_unnorm = self._unnormalize_actions(action_venv)
obs_venv, reward, ... = self.venv.step(action_venv_unnorm)
```

### 10.2 Critic Preprocessing

MPI has special critic preprocessing for asymmetric actor-critic:

```python
def _preprocess_obs_for_critic(self, obs):
    if self.critic_type == "mlp":
        # Use full_state (privileged info)
        full_state = obs["full_state"][:, -1]  # Latest only
        return {"state": full_state}
    else:
        # MPI/ViT: extract latest frames
        return {
            "rgb": obs["rgb"][:, -self.critic_img_cond_steps:],
            "state": obs["state"][:, -self.critic_n_obs_steps:]
        }
```

### 10.3 Custom Metrics

MPI tracks additional metrics:
- `avg_pieces_per_hour`
- `avg_task_completion`
- `avg_success_duration` / `avg_fail_duration`
- `n_successful_grasps` / `n_failed_grasps` / `n_failed_drops`

---

## 11. Potential Experiments

Based on the differences identified, here are experiments to try:

### 11.1 Reward Structure Experiments

1. **Try sparse reward like Robomimic:**
   - Remove dense shaping, use only terminal rewards
   - Keep step penalty but remove distance-based shaping
   
2. **Try no early termination:**
   - Continue episode after success (like Robomimic)
   - Give +1 for remaining steps after removing all boxes
   
3. **Adjust reward scales:**
   - Success reward vs step penalty ratio
   - Event reward magnitudes

### 11.2 Trajectory and Horizon Experiments

1. **Reduce action horizon:**
   - Try `act_steps: 4` like Robomimic
   - Shorter `horizon_steps: 8` 
   
2. **More policy steps per episode:**
   - Increase max_episode_steps or reduce act_steps

### 11.3 Architecture Experiments

1. **Smaller policy network:**
   - Use 2-level UNet like Robomimic
   - Reduce diffusion step embed dim
   
2. **Critic architecture:**
   - Try simpler MLP critic like Robomimic
   - Use same observation as policy

### 11.4 Diffusion Experiments

1. **Fewer denoising steps:**
   - Try `denoising_steps: 20` like Robomimic
   - Adjust `ft_denoising_steps` accordingly
   
2. **Higher sampling std:**
   - `min_sampling_denoising_std: 0.1` like Robomimic

### 11.5 Training Configuration

1. **Larger batch without accumulation:**
   - `batch_size: 10000, grad_accumulate: 1`
   
2. **More samples per iteration:**
   - Increase `n_steps` to 400

### 11.6 Observation Experiments

1. **Single observation step:**
   - `cond_steps: 1` like Robomimic
   - Remove history from policy and critic

---

## Summary of Critical Differences

| Category | Most Likely Impact | Robomimic (Simple) | Robomimic (Transport) | MPI 4-Box |
|----------|-------------------|--------------------|-----------------------|-----------|
| **Episode structure** | High | Fixed length, no early term | Fixed length, no early term | Variable, early termination |
| **Reward density** | High | Sparse (0/1) | Sparse (0/1) | Dense multi-component |
| **Action horizon** | High | 4 steps | 8 steps | 20 steps |
| **Prediction horizon** | Medium | 4 steps | 16 steps | 32 steps |
| **Max policy steps** | High | 75-100 | 100 | 40 |
| **Task complexity** | High | Single goal | Bimanual coordination | Multi-stage per box |
| **Policy network dim** | Medium | 40-64 | 64 | 256-1024 |
| **Obs dimensionality** | Medium | 19-23D state | 59D state | 6D state + depth image |
| **Action dimensionality** | Medium | 7D EE | 14D EE (two arms) | 5D joints |
| **Denoising steps** | Medium | 20 | 20 | 100 |
| **Observation history** | Low | 1 step | 1 step | 2 steps |

**Key Insight:** Transport is the closest Robomimic task to MPI in terms of horizon ratios and complexity, making it a better baseline for comparison than simpler tasks like Square.
