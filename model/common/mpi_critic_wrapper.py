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
    
    Key differences from MPICriticWrapper:
    1. The value network is initialized fresh (random weights)
    2. Uses the policy normalizer for observation normalization (set during init)
    3. All parameters are trainable from the start
    4. Does NOT unnormalize value outputs (PPO works with raw values)
    
    The fresh critic will learn value estimates during PPO training, using the
    `n_critic_warmup_itr` config to freeze the actor while the critic catches up.
    
    Note: Unlike pretrained MPI critics that unnormalize values via predict_value(),
    this wrapper returns raw value predictions. This is correct for PPO training
    where value unnormalization is not needed (and the policy normalizer doesn't
    have a "value" key anyway).
    """
    
    def __init__(
        self,
        mpi_value_network: nn.Module,
        n_obs_steps: int = 1,
        img_cond_steps: int = 1,
    ):
        """
        Args:
            mpi_value_network: Fresh MPI value network (e.g., ResNet18TVTruck2dValueNetwork)
                               Should already have normalizer set via set_normalizer()
            n_obs_steps: Number of state observation steps to use (typically 1 for critic)
            img_cond_steps: Number of image frames to use (typically 1)
        """
        super().__init__()
        self.value_network = mpi_value_network
        self.n_obs_steps = n_obs_steps
        self.img_cond_steps = img_cond_steps
        
        log.info(
            f"MPICriticWrapperFresh initialized with n_obs_steps={n_obs_steps}, "
            f"img_cond_steps={img_cond_steps} (fresh critic, trained from scratch)"
        )
    
    def forward(
        self,
        cond: dict,
        no_augment: bool = False,
    ) -> torch.Tensor:
        """
        Compute value estimates from observations.
        
        Normalizes observations using the policy normalizer, calls forward,
        and returns RAW value predictions (no unnormalization).
        
        Args:
            cond: Dict with observation history
                - 'state': [B, To, Do] state observations
                - 'rgb': [B, To, C, H, W] image observations
            no_augment: Whether to skip augmentation (passed for API compatibility)
        
        Returns:
            values: [B, 1] raw value estimates (not unnormalized)
        """
        B = cond['rgb'].shape[0] if 'rgb' in cond else cond['state'].shape[0]
        
        # Extract latest frame(s) from history
        # Handle RGB/images
        if 'rgb' in cond:
            rgb = cond['rgb']
            if rgb.dim() == 5:  # [B, T, C, H, W]
                rgb = rgb[:, -self.img_cond_steps:]
                if self.img_cond_steps == 1:
                    rgb = rgb[:, 0]  # [B, C, H, W]
                else:
                    rgb = rgb.reshape(B, -1, rgb.shape[-2], rgb.shape[-1])
            img_obs = rgb
        else:
            raise ValueError("MPICriticWrapperFresh requires 'rgb' in cond dict")
        
        # Handle state
        if 'state' in cond:
            state = cond['state']
            if state.dim() == 3:  # [B, T, D]
                state = state[:, -self.n_obs_steps:]
                if self.n_obs_steps == 1:
                    state = state[:, 0]  # [B, D]
                else:
                    state = state.reshape(B, -1)
            vec_obs = state
        else:
            raise ValueError("MPICriticWrapperFresh requires 'state' in cond dict")
        
        # For fresh critic: normalize inputs, call forward, return raw values
        # We DON'T use predict_value() because it tries to unnormalize the output
        # using a "value" key that doesn't exist in the policy normalizer.
        # PPO doesn't need unnormalized values anyway.
        
        if hasattr(self.value_network, 'normalizer'):
            # Normalize inputs using the value network's normalizer (policy normalizer)
            data_n = self.value_network.normalizer.normalize({
                'images': img_obs,
                'robot_state': vec_obs
            })
            img_norm = data_n['images']
            vec_norm = data_n['robot_state']
        else:
            # No normalizer - use raw inputs
            img_norm = img_obs
            vec_norm = vec_obs
        
        # Call forward directly (returns raw values, no unnormalization)
        values = self.value_network(img_norm, vec_norm)
        
        return values
