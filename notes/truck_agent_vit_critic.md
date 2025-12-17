# Plan: Using DPPO's ViTCritic for Truck2D Task

## Overview

This document outlines the plan to replace the MPI ResNet18-based critic with DPPO's standard `ViTCritic` from `model/common/critic.py`. The goal is to use a simpler, proven architecture that the DPPO codebase uses successfully for other image-based tasks.

## Current Architecture

**MPI Critic (ResNet18TVTruck2dValueNetwork)**:
- ResNet-18 backbone (~11M parameters)
- Trained from scratch via `MPICriticWrapperFresh`
- Handles its own normalization internally
- Uses `_preprocess_obs_for_critic()` in agent to extract single-frame observations

**Issues**:
- Large network is hard to train from scratch during PPO
- Explained variance stays ~0, suggesting value collapse
- Complex normalization flow between MPI and DPPO conventions

## Target Architecture

**DPPO ViTCritic**:
- Lightweight ViT backbone (~500K parameters depending on config)
- `SpatialEmb` compression layer for efficient feature extraction
- MLP head (e.g., `[256, 256, 256]`)
- Proven to work for DPPO image-based tasks

## Key Differences to Address

### 1. Image Format

| Property | DPPO Standard | Truck2D |
|----------|---------------|---------|
| Channels | 3 (RGB) | 1 (grayscale) |
| Height | 96 | 71 |
| Width | 96 | 192 |
| Value range | [0, 255] uint8 | [0, 1] float32 |

**Solution**: 
- Configure `VitEncoder` with `num_channel=1` (or `img_cond_steps` for history)
- Configure `img_h=71`, `img_w=192`
- Convert images from [0,1] to [0,255] before passing to ViTCritic (DPPO's VitEncoder divides by 255 internally)

### 2. Image Normalization

DPPO's `VitEncoder.forward()` does:
```python
obs = obs / 255.0 - 0.5
```

MPI's truck2d env returns images in [0, 1]. We need to convert:
```python
# In agent, before passing to critic:
rgb_for_critic = cond['rgb'] * 255.0  # Convert [0,1] -> [0,255]
```

### 3. Observation History Handling

DPPO's `ViTCritic` handles history by:
1. Taking recent `img_cond_steps` frames from `cond['rgb']`
2. Concatenating in channel dimension: `(B, T, C, H, W) -> (B, T*C, H, W)`
3. Processing through backbone

For truck2d:
- `cond_steps=2` (policy uses 2-step history)
- Critic can use `img_cond_steps=1` (single frame) or `img_cond_steps=2` (history)
- State history is flattened: `(B, T, D) -> (B, T*D)`

### 4. Non-Square Images

DPPO's `VitEncoder` uses `PatchEmbed2` which computes patch counts as:
```python
H1 = math.ceil((img_h - 8) / 4) + 1  # After first conv
W1 = math.ceil((img_w - 8) / 4) + 1
H2 = math.ceil((H1 - 3) / 2) + 1     # After second conv
W2 = math.ceil((W1 - 3) / 2) + 1
num_patch = H2 * W2
```

For truck2d (71x192):
- H1 = ceil((71-8)/4) + 1 = 17
- W1 = ceil((192-8)/4) + 1 = 47
- H2 = ceil((17-3)/2) + 1 = 8
- W2 = ceil((47-3)/2) + 1 = 23
- num_patch = 8 * 23 = 184 patches

This is larger than the 144 patches (12x12) for standard 96x96 images, but should work fine.

### 5. RandomShiftsAug Assumption

`RandomShiftsAug` in `model/common/modules.py` has `assert h == w` (line 50). This will fail for truck2d's non-square images.

**Solution**: Set `augment=False` for the critic (recommended for PPO anyway to reduce variance).

## Design Decision: Where Does History Extraction Happen?

**Problem**: The current MPI critic uses `_preprocess_obs_for_critic()` in the agent to extract latest frames. If we add a DPPO critic, where should extraction happen?

**Options Considered**:
1. **Agent handles extraction for all critics** ✅ CHOSEN
2. Each wrapper handles its own extraction (duplicated logic)
3. Mixed approach (inconsistent, bad design)

**Chosen Approach**: Agent's `_preprocess_obs_for_critic()` handles history extraction for ALL critics. Wrappers only handle critic-specific transformations:
- **MPI wrapper**: Input normalization via LinearNormalizer
- **DPPO wrapper**: Image range conversion [0,1] → [0,255]

**Rationale**:
- Single location for extraction logic (DRY principle)
- Wrappers have clear, single responsibilities
- Consistent interface: all critics receive pre-extracted observations
- Easy to reason about data flow

## Implementation Plan

### Phase 1: Refactor Existing Code for Consistency

Before adding ViTCritic, fix the **double extraction bug** in the current architecture.

#### Current Bug: Double Extraction

The agent and wrapper both do extraction:
1. Agent's `_preprocess_obs_for_critic()`: `[B, T, C, H, W]` → `[B, critic_img_cond_steps, C, H, W]`
2. `MPICriticWrapperFresh.forward()`: extracts again using `self.img_cond_steps`

This is redundant and error-prone if the values don't match.

#### 1.1 Simplify `MPICriticWrapperFresh`

Remove extraction logic. The wrapper should only:
1. Squeeze time dimension if single frame (the data is already extracted)
2. Normalize observations
3. Call the value network

```python
class MPICriticWrapperFresh(nn.Module):
    """
    Wrapper for fresh (untrained) MPI value network.
    
    Responsibilities:
    - Squeeze time dimension if observations are single-frame
    - Normalize observations using policy normalizer
    - Call forward() and return raw values
    
    Does NOT handle history extraction (agent does that via _preprocess_obs_for_critic).
    """
    
    def __init__(self, mpi_value_network: nn.Module):
        super().__init__()
        self.value_network = mpi_value_network
        
        log.info("MPICriticWrapperFresh initialized (extraction handled by agent)")
    
    def forward(self, cond: dict, no_augment: bool = False) -> torch.Tensor:
        """
        Args:
            cond: Dict with PRE-EXTRACTED observations from agent:
                - 'rgb': (B, critic_img_cond_steps, C, H, W) 
                - 'state': (B, critic_n_obs_steps, D)
        """
        B = cond['rgb'].shape[0]
        rgb = cond['rgb']
        state = cond['state']
        
        # Squeeze time dimension if single frame
        # Agent extracts to [B, N, ...], we need [B, ...] for MPI value network
        if rgb.dim() == 5 and rgb.shape[1] == 1:
            rgb = rgb[:, 0]  # (B, C, H, W)
        elif rgb.dim() == 5:
            # Multiple frames - flatten into channels (if network supports it)
            rgb = rgb.reshape(B, -1, rgb.shape[-2], rgb.shape[-1])
            
        if state.dim() == 3 and state.shape[1] == 1:
            state = state[:, 0]  # (B, D)
        elif state.dim() == 3:
            # Multiple timesteps - flatten into features
            state = state.reshape(B, -1)
        
        # Normalize using value network's normalizer
        if hasattr(self.value_network, 'normalizer'):
            data_n = self.value_network.normalizer.normalize({
                'images': rgb,
                'robot_state': state
            })
            rgb = data_n['images']
            state = data_n['robot_state']
        
        return self.value_network(rgb, state)
```

**Key changes**:
- Removed `n_obs_steps` and `img_cond_steps` from `__init__` (no longer needed)
- Removed extraction logic from `forward()`
- Added squeeze logic for single-frame case (agent provides `[B, 1, ...]`, we convert to `[B, ...]`)

### Phase 2: Add DPPO ViTCritic Support

#### 2.1 Create `DPPOViTCriticWrapper` (`model/common/mpi_critic_wrapper.py`)

Thin wrapper that handles image range conversion. Note that `ViTCritic` internally handles:
- Time dimension: concatenates frames in channel dimension via `einops.rearrange`
- State flattening: `state.view(B, -1)`

So the wrapper just needs to convert image range:

```python
class DPPOViTCriticWrapper(nn.Module):
    """
    Wrapper for DPPO's ViTCritic to work with truck2d's observation format.
    
    Responsibilities:
    - Convert image range from [0,1] to [0,255] (DPPO's VitEncoder expects this)
    - Pass through to ViTCritic.forward()
    
    Does NOT handle history extraction (agent does that via _preprocess_obs_for_critic).
    ViTCritic internally handles time dimension (channel concatenation) and state flattening.
    """
    
    def __init__(self, vit_critic: nn.Module):
        super().__init__()
        self.vit_critic = vit_critic
        
        log.info("DPPOViTCriticWrapper initialized (extraction handled by agent)")
    
    def forward(self, cond: dict, no_augment: bool = True) -> torch.Tensor:
        """
        Args:
            cond: Dict with PRE-EXTRACTED observations from agent:
                - 'rgb': (B, critic_img_cond_steps, C, H, W) in [0,1]
                - 'state': (B, critic_n_obs_steps, D)
                
        Note: ViTCritic internally handles:
            - Channel concatenation: (B, T, C, H, W) -> (B, T*C, H, W)
            - State flattening: (B, T, D) -> (B, T*D)
        """
        # Convert image range: [0,1] -> [0,255] for VitEncoder
        # (VitEncoder.forward does: obs = obs / 255.0 - 0.5)
        critic_cond = {
            'rgb': cond['rgb'] * 255.0,
            'state': cond['state'],
        }
        
        return self.vit_critic(critic_cond, no_augment=no_augment)
```

#### 2.2 Modify `MPIPPODiffusion.from_mpi_checkpoints()`

Add `critic_type` parameter. Note that wrappers no longer need `n_obs_steps`/`img_cond_steps` since extraction is handled by the agent:

```python
@classmethod
def from_mpi_checkpoints(
    cls,
    # ... existing args ...
    critic_type: str = 'mpi',  # NEW: 'mpi' or 'vit'
    # Note: critic_n_obs_steps and critic_img_cond_steps are used by the AGENT,
    # not by the wrappers. The wrappers receive pre-extracted observations.
    # ...
):
    from model.common.mpi_critic_wrapper import MPICriticWrapperFresh, DPPOViTCriticWrapper
    
    # ...
    
    if not use_pretrained_critic:
        if critic_type == 'vit':
            log.info("Creating fresh DPPO ViTCritic")
            vit_critic = hydra.utils.instantiate(critic)
            vit_critic = vit_critic.to(device)
            # Wrapper only handles image range conversion
            wrapped_critic = DPPOViTCriticWrapper(vit_critic)
            
        elif critic_type == 'mpi':
            log.info("Creating fresh MPI ResNet critic")
            fresh_value_network = hydra.utils.instantiate(critic)
            fresh_value_network = fresh_value_network.to(device)
            # Set normalizer for observation normalization
            if hasattr(fresh_value_network, 'set_normalizer'):
                policy_normalizer = torch.load(
                    policy_normalizer_path, map_location=device, weights_only=False
                )
                fresh_value_network.set_normalizer(policy_normalizer)
                log.info("Set policy normalizer on fresh MPI critic")
            # Wrapper handles squeeze + normalization
            wrapped_critic = MPICriticWrapperFresh(fresh_value_network)
            
        else:
            raise ValueError(f"Unknown critic_type: {critic_type}. Must be 'mpi' or 'vit'.")
```

#### 2.3 Agent: Keep `_preprocess_obs_for_critic()` Unchanged

The existing `_preprocess_obs_for_critic()` already extracts latest frames based on `critic_n_obs_steps` and `critic_img_cond_steps`. This works for BOTH critic types:

```python
def _preprocess_obs_for_critic(self, obs):
    """
    Extract latest frames for critic (works for both MPI and DPPO critics).
    
    Policy may use 2-step history, but critic typically uses 1 step.
    This function extracts the appropriate number of frames based on config.
    """
    critic_obs = {}
    
    if 'rgb' in obs:
        rgb = obs['rgb']
        if isinstance(rgb, torch.Tensor) and rgb.dim() == 5:
            critic_obs['rgb'] = rgb[:, -self.critic_img_cond_steps:]
        else:
            critic_obs['rgb'] = rgb
    
    if 'state' in obs:
        state = obs['state']
        if isinstance(state, torch.Tensor) and state.dim() == 3:
            critic_obs['state'] = state[:, -self.critic_n_obs_steps:]
        else:
            critic_obs['state'] = state
    
    return critic_obs
```

**No changes needed here** - the existing implementation is already correct and general!

### Phase 3: Config Changes

#### 3.1 New Config for ViT Critic

Create `cfg/truck_2d/ft_ppo_mpi_truck_2d_vit_critic.yaml`:

```yaml
# ... (copy base config) ...

# Critic configuration
critic_n_obs_steps: 1      # State history for critic
critic_img_cond_steps: 1   # Image history for critic

model:
  _target_: model.diffusion.mpi_ppo_diffusion.MPIPPODiffusion.from_mpi_checkpoints
  
  policy_checkpoint_path: ${base_policy_path}
  policy_normalizer_path: ${policy_normalizer_path}
  
  # Use DPPO's ViTCritic
  use_pretrained_critic: false
  critic_type: vit  # NEW
  critic:
    _target_: model.common.critic.ViTCritic
    backbone:
      _target_: model.common.vit.VitEncoder
      obs_shape: ${shape_meta.obs.rgb.shape}
      num_channel: ${critic_img_cond_steps}  # 1 channel * img_cond_steps
      img_h: 71
      img_w: 192
      cfg:
        patch_size: 8
        depth: 1
        embed_dim: 128
        num_heads: 4
        embed_style: embed2
        embed_norm: 0
    img_cond_steps: ${critic_img_cond_steps}
    spatial_emb: 128
    mlp_dims: [256, 256, 256]
    activation_type: Mish
    residual_style: true
    cond_dim: ${eval:'${obs_dim} * ${critic_n_obs_steps}'}
    augment: false  # REQUIRED: Non-square images break RandomShiftsAug
  
  # Other model params (unchanged)
  ft_denoising_steps: ${ft_denoising_steps}
  # ...
```

#### 3.2 Existing MPI Config (Minimal Changes)

Just add `critic_type: mpi` to make it explicit:

```yaml
model:
  # ...
  use_pretrained_critic: false
  critic_type: mpi  # Explicit (default)
  critic:
    _target_: value_network.models.resnet18_tv_truck_2d_value_network.ResNet18TVTruck2dValueNetwork
    # ...
```

## Files to Modify

### 1. `model/common/mpi_critic_wrapper.py`

**Changes to `MPICriticWrapperFresh`**:
- Remove `n_obs_steps` and `img_cond_steps` from `__init__` signature
- Remove extraction logic from `forward()` (lines 289-314 in current code)
- Keep squeeze logic for single-frame case
- Keep normalization logic

**Add new class `DPPOViTCriticWrapper`**:
- Simple wrapper that converts [0,1] → [0,255]
- Passes through to `ViTCritic.forward()`

### 2. `model/diffusion/mpi_ppo_diffusion.py`

**Changes to `from_mpi_checkpoints()`**:
- Add `critic_type: str = 'mpi'` parameter
- Add branching logic for 'mpi' vs 'vit'
- Remove `n_obs_steps`/`img_cond_steps` from wrapper instantiation (no longer needed)
- For 'vit': instantiate `ViTCritic` + wrap with `DPPOViTCriticWrapper`
- For 'mpi': use existing logic but with simplified wrapper

### 3. `cfg/truck_2d/ft_ppo_mpi_truck_2d_vit_critic.yaml` (NEW FILE)

New config file for ViTCritic experiments.

### 4. `cfg/truck_2d/ft_ppo_mpi_truck_2d_single_box_fresh_critic.yaml` (OPTIONAL)

Add explicit `critic_type: mpi` for clarity/documentation.

### Agent Changes: NONE REQUIRED ✓

The existing `_preprocess_obs_for_critic()` in `train_ppo_diffusion_truck2d_agent.py` already:
- Extracts based on `self.critic_img_cond_steps` and `self.critic_n_obs_steps`
- Works correctly for both critic types
- No code changes needed!

## Testing Plan

1. **Unit test**: Verify `DPPOViTCriticWrapper` produces correct output shapes
2. **Integration test**: Run training for 10 iterations, verify:
   - No shape errors
   - Critic loss decreases
   - Explained variance trends upward (not stuck at 0)
3. **Comparison**: Run side-by-side with MPI critic, compare learning curves

## Alternatives Considered

### A. Each Wrapper Handles Its Own Extraction

Have extraction logic in both `MPICriticWrapperFresh` and `DPPOViTCriticWrapper`.

**Rejected**: Duplicated logic, violates DRY principle.

### B. Mixed Approach (Agent for MPI, Wrapper for DPPO)

Let the agent preprocess for MPI but not for DPPO.

**Rejected**: Inconsistent design, confusing to maintain.

### C. Modify VitEncoder to Accept [0,1] Images

Change `VitEncoder.forward()` to handle both ranges.

**Rejected**: Modifies shared code, could affect other tasks.

### D. Modify RandomShiftsAug for Non-Square Images

Fix the `assert h == w` constraint.

**Deferred**: Augmentation for critic is not recommended for PPO anyway due to added variance.

## Summary

### Bug Fix: Double Extraction

The current code has a subtle bug where extraction happens twice:
1. Agent's `_preprocess_obs_for_critic()` extracts latest N frames
2. `MPICriticWrapperFresh` extracts again internally

This is redundant and error-prone. The fix: remove extraction from wrappers.

### Key Design Principles

1. **Agent handles extraction for ALL critics**: `_preprocess_obs_for_critic()` extracts latest frames based on `critic_n_obs_steps` and `critic_img_cond_steps`. This code already exists and works unchanged.

2. **Wrappers have single responsibilities**:
   - `MPICriticWrapperFresh`: Squeeze time dim + normalize via LinearNormalizer
   - `DPPOViTCriticWrapper`: Convert [0,1]→[0,255] (ViTCritic handles time dim internally)

3. **Clean interface**: Both wrappers receive pre-extracted observations with shape:
   - `rgb`: (B, critic_img_cond_steps, C, H, W)
   - `state`: (B, critic_n_obs_steps, D)

4. **Config-driven**: Switch between critics by changing `critic_type: mpi` or `critic_type: vit`

### Benefits

- Fixes double-extraction bug
- Centralizes extraction logic (DRY principle)
- Minimal code changes (2 files + 1 new config)
- No changes needed to agent code
- Easy to add more critic types in the future
- Clear separation of concerns

## Appendix: Parameter Count Comparison

| Critic | Approximate Parameters |
|--------|----------------------|
| ResNet18TVTruck2dValueNetwork | ~11.2M |
| ViTCritic (embed_dim=128, depth=1, mlp=[256,256,256]) | ~0.5M |

The ViTCritic is ~20x smaller, which should be much easier to train from scratch during PPO.
