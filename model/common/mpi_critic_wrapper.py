"""
MPI Critic Wrapper for DPPO integration.

This module provides a wrapper class to adapt MPI's ResNet18TVTruck2dValueNetwork
(or similar value networks) to DPPO's critic interface.

DPPO's critic interface:
    critic(cond, no_augment=False) where cond = {'state': [B,T,D], 'rgb': [B,T,C,H,W]}
    
MPI's value network interface:
    forward(img_norm, vec_norm) where img_norm=[B,C,H,W], vec_norm=[B,D]
    - Takes single frame observations (not history)
    - Has its own normalizer for input normalization
"""

import logging
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class MPICriticWrapper(nn.Module):
    """
    Wraps MPI's value network to provide DPPO-compatible critic interface.
    
    Key differences handled:
    1. DPPO passes observation history, MPI value network uses single frame
    2. DPPO passes dict with 'state'/'rgb', MPI expects separate img/vec tensors
    3. MPI value network has its own normalizer (handled internally)
    
    The wrapper extracts the latest observation from history and calls the
    MPI value network with properly formatted inputs.
    """
    
    def __init__(
        self,
        mpi_value_network: nn.Module,
        normalizer=None,
        n_obs_steps: int = 1,
        img_cond_steps: int = 1,
    ):
        """
        Args:
            mpi_value_network: MPI's value network (e.g., ResNet18TVTruck2dValueNetwork)
            normalizer: Optional normalizer for observations. If None, uses the
                       value network's internal normalizer.
            n_obs_steps: Number of observation steps to use (typically 1 for critic)
            img_cond_steps: Number of image frames to use (typically 1)
        """
        super().__init__()
        self.value_network = mpi_value_network
        self.external_normalizer = normalizer
        self.n_obs_steps = n_obs_steps
        self.img_cond_steps = img_cond_steps
        
        log.info(f"MPICriticWrapper initialized with n_obs_steps={n_obs_steps}, "
                 f"img_cond_steps={img_cond_steps}")
    
    def forward(
        self,
        cond: dict,
        no_augment: bool = False,
    ) -> torch.Tensor:
        """
        Compute value estimates from observations.
        
        Args:
            cond: Dict with observation history
                - 'state': [B, To, Do] state observations
                - 'rgb': [B, To, C, H, W] image observations
            no_augment: Whether to skip augmentation (passed for API compatibility,
                       but MPI value network doesn't use augmentation)
        
        Returns:
            values: [B, 1] value estimates
        """
        B = cond['rgb'].shape[0] if 'rgb' in cond else cond['state'].shape[0]
        
        # Extract latest frame(s) from history
        # MPI value network typically uses single frame
        
        # Handle RGB/images
        if 'rgb' in cond:
            rgb = cond['rgb']
            # Take latest img_cond_steps frames
            if rgb.dim() == 5:  # [B, T, C, H, W]
                rgb = rgb[:, -self.img_cond_steps:]  # [B, img_cond_steps, C, H, W]
                # If single frame, squeeze the time dimension
                if self.img_cond_steps == 1:
                    rgb = rgb[:, 0]  # [B, C, H, W]
                else:
                    # Stack frames in channel dimension
                    rgb = rgb.reshape(B, -1, rgb.shape[-2], rgb.shape[-1])
            img_obs = rgb
        else:
            raise ValueError("MPICriticWrapper requires 'rgb' in cond dict")
        
        # Handle state
        if 'state' in cond:
            state = cond['state']
            # Take latest n_obs_steps
            if state.dim() == 3:  # [B, T, D]
                state = state[:, -self.n_obs_steps:]  # [B, n_obs_steps, D]
                # If single step, squeeze
                if self.n_obs_steps == 1:
                    state = state[:, 0]  # [B, D]
                else:
                    # Flatten time into feature dimension
                    state = state.reshape(B, -1)
            vec_obs = state
        else:
            raise ValueError("MPICriticWrapper requires 'state' in cond dict")
        
        # Call MPI value network
        # The value network handles its own normalization internally
        # via its normalizer and forward/predict_value methods
        
        # Check if value network has predict_value method (handles normalization)
        if hasattr(self.value_network, 'predict_value'):
            # Use predict_value which handles normalization internally
            values = self.value_network.predict_value(img_obs, vec_obs)
        else:
            # Direct forward call - may need manual normalization
            if self.external_normalizer is not None:
                # Normalize using external normalizer
                data_n = self.external_normalizer.normalize({
                    'images': img_obs,
                    'robot_state': vec_obs
                })
                img_norm = data_n['images']
                vec_norm = data_n['robot_state']
            elif hasattr(self.value_network, 'normalizer'):
                # Use value network's normalizer
                data_n = self.value_network.normalizer.normalize({
                    'images': img_obs,
                    'robot_state': vec_obs
                })
                img_norm = data_n['images']
                vec_norm = data_n['robot_state']
            else:
                # No normalization
                img_norm = img_obs
                vec_norm = vec_obs
            
            values = self.value_network(img_norm, vec_norm)
        
        return values
    
    def set_normalizer(self, normalizer):
        """
        Set the normalizer for the value network.
        
        Args:
            normalizer: LinearNormalizer instance
        """
        if hasattr(self.value_network, 'set_normalizer'):
            self.value_network.set_normalizer(normalizer)
        else:
            self.external_normalizer = normalizer
        log.info("Set normalizer for MPICriticWrapper")


class MPICriticWrapperDirect(nn.Module):
    """
    Direct wrapper that calls MPI value network's forward method directly.
    
    Use this when the value network doesn't need external normalization
    (i.e., observations are already normalized or the network handles it).
    """
    
    def __init__(
        self,
        mpi_value_network: nn.Module,
        n_obs_steps: int = 1,
        img_cond_steps: int = 1,
    ):
        """
        Args:
            mpi_value_network: MPI's value network
            n_obs_steps: Number of state observation steps to use
            img_cond_steps: Number of image frames to use
        """
        super().__init__()
        self.value_network = mpi_value_network
        self.n_obs_steps = n_obs_steps
        self.img_cond_steps = img_cond_steps
    
    def forward(
        self,
        cond: dict,
        no_augment: bool = False,
    ) -> torch.Tensor:
        """
        Compute value estimates from observations.
        
        Assumes observations are already normalized appropriately.
        """
        B = cond['rgb'].shape[0] if 'rgb' in cond else cond['state'].shape[0]
        
        # Extract latest frames
        rgb = cond['rgb']
        if rgb.dim() == 5:
            rgb = rgb[:, -self.img_cond_steps:]
            if self.img_cond_steps == 1:
                rgb = rgb[:, 0]
            else:
                rgb = rgb.reshape(B, -1, rgb.shape[-2], rgb.shape[-1])
        
        state = cond['state']
        if state.dim() == 3:
            state = state[:, -self.n_obs_steps:]
            if self.n_obs_steps == 1:
                state = state[:, 0]
            else:
                state = state.reshape(B, -1)
        
        # Direct forward call
        return self.value_network(rgb, state)


class MPICriticWrapperFresh(nn.Module):
    """
    Wrapper for a FRESH (not pretrained) MPI value network.
    
    This is used when training the critic from scratch during PPO fine-tuning,
    instead of loading a pretrained critic checkpoint.
    
    Responsibilities:
    - Squeeze time dimension if observations are single-frame
    - Normalize observations using policy normalizer
    - Call forward() and return raw values
    
    Does NOT handle history extraction - that is done by the agent via
    `_preprocess_obs_for_critic()`. This wrapper receives PRE-EXTRACTED observations.
    
    Note: Unlike pretrained MPI critics that unnormalize values via predict_value(),
    this wrapper returns raw value predictions. This is correct for PPO training
    where value unnormalization is not needed.
    """
    
    def __init__(self, mpi_value_network: nn.Module):
        """
        Args:
            mpi_value_network: Fresh MPI value network (e.g., ResNet18TVTruck2dValueNetwork)
                               Should already have normalizer set via set_normalizer()
        """
        super().__init__()
        self.value_network = mpi_value_network
        
        log.info("MPICriticWrapperFresh initialized (extraction handled by agent)")
    
    def forward(
        self,
        cond: dict,
        no_augment: bool = False,
    ) -> torch.Tensor:
        """
        Compute value estimates from PRE-EXTRACTED observations.
        
        Args:
            cond: Dict with PRE-EXTRACTED observations from agent:
                - 'rgb': (B, critic_img_cond_steps, C, H, W) 
                - 'state': (B, critic_n_obs_steps, D)
            no_augment: Whether to skip augmentation (passed for API compatibility)
        
        Returns:
            values: [B, 1] raw value estimates (not unnormalized)
        """
        B = cond['rgb'].shape[0] if 'rgb' in cond else cond['state'].shape[0]
        rgb = cond['rgb']
        state = cond['state']
        
        # Squeeze time dimension if single frame
        # Agent extracts to [B, N, ...], MPI value network expects [B, ...]
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
        
        # Normalize inputs using the value network's normalizer (policy normalizer)
        if hasattr(self.value_network, 'normalizer'):
            data_n = self.value_network.normalizer.normalize({
                'images': rgb,
                'robot_state': state
            })
            img_norm = data_n['images']
            vec_norm = data_n['robot_state']
        else:
            # No normalizer - use raw inputs
            img_norm = rgb
            vec_norm = state
        
        # Call forward directly (returns raw values, no unnormalization)
        values = self.value_network(img_norm, vec_norm)
        
        return values


class DPPOViTCriticWrapper(nn.Module):
    """
    Wrapper for DPPO's ViTCritic to work with truck2d's observation format.
    
    Responsibilities:
    - Convert image range from [0,1] to [0,255] (DPPO's VitEncoder expects this)
    - Pass through to ViTCritic.forward()
    
    Does NOT handle history extraction - that is done by the agent via
    `_preprocess_obs_for_critic()`. This wrapper receives PRE-EXTRACTED observations.
    
    Note: ViTCritic internally handles:
    - Channel concatenation: (B, T, C, H, W) -> (B, T*C, H, W) via einops
    - State flattening: (B, T, D) -> (B, T*D)
    """
    
    def __init__(self, vit_critic: nn.Module):
        """
        Args:
            vit_critic: DPPO's ViTCritic instance
        """
        super().__init__()
        self.vit_critic = vit_critic
        
        log.info("DPPOViTCriticWrapper initialized (extraction handled by agent)")
    
    def forward(self, cond: dict, no_augment: bool = True) -> torch.Tensor:
        """
        Compute value estimates from PRE-EXTRACTED observations.
        
        Args:
            cond: Dict with PRE-EXTRACTED observations from agent:
                - 'rgb': (B, critic_img_cond_steps, C, H, W) in [0,1]
                - 'state': (B, critic_n_obs_steps, D)
            no_augment: Whether to skip augmentation (default True for PPO)
                
        Note: ViTCritic internally handles:
            - Channel concatenation: (B, T, C, H, W) -> (B, T*C, H, W)
            - State flattening: (B, T, D) -> (B, T*D)
        
        Returns:
            values: [B, 1] value estimates
        """
        # Convert image range: [0,1] -> [0,255] for VitEncoder
        # (VitEncoder.forward does: obs = obs / 255.0 - 0.5)
        critic_cond = {
            'rgb': cond['rgb'] * 255.0,
            'state': cond['state'],
        }
        
        return self.vit_critic(critic_cond, no_augment=no_augment)
