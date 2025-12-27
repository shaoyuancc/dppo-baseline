"""
MPI-DPPO Integration: PPO Diffusion with MPI's policy architecture.

This module provides wrapper classes to adapt MPI's DiffusionUnetHybridImageTargetedPolicy
to DPPO's training interface, enabling PPO-based finetuning of MPI checkpoints.

Supports both DDPM and DDIM sampling via the `use_ddim` flag, matching original DPPO behavior.

Key components:
- MPIPolicyActorWrapper: Wraps MPI's obs_encoder + UNet for DPPO-compatible interface
- MPIPPODiffusion: DPPO model using MPI's architecture with DDPM/DDIM sampling

Usage in config:
    model:
      _target_: model.diffusion.mpi_ppo_diffusion.MPIPPODiffusion.from_mpi_checkpoints
      policy_checkpoint_path: /path/to/policy.ckpt
      policy_normalizer_path: /path/to/normalizer.pt
      critic_checkpoint_path: /path/to/critic.pt
      critic_normalizer_path: /path/to/critic_normalizer.pt
      use_ddim: false  # Set to true for DDIM sampling
      ddim_steps: 5    # Required when use_ddim=true
      ...
"""

import copy
import logging
import torch
import torch.nn as nn
import hydra
from collections import namedtuple

log = logging.getLogger(__name__)

from model.diffusion.sampling import make_timesteps, extract, cosine_beta_schedule

Sample = namedtuple("Sample", "trajectories chains")


class MPIPolicyActorWrapper(nn.Module):
    """
    Wraps MPI's observation encoder and UNet to provide DPPO-compatible interface.
    
    DPPO calls: actor(x, t, cond=cond) where cond = {'state': [B,T,D], 'rgb': [B,T,C,H,W]}
    MPI calls: model(sample, timestep, global_cond=encoded_obs)
    
    This wrapper handles:
    1. Extracting observations from cond dict
    2. Normalizing observations using MPI's normalizer
    3. Encoding observations using MPI's obs_encoder
    4. Calling MPI's UNet model
    """
    
    def __init__(
        self,
        obs_encoder: nn.Module,
        unet_model: nn.Module,
        normalizer,
        n_obs_steps: int = 2,
        obs_feature_dim: int = None,
        project_obs_embedding: bool = False,
        obs_embedding_projector: nn.Module = None,
    ):
        """
        Args:
            obs_encoder: MPI's observation encoder (from RoboMimic)
            unet_model: MPI's ConditionalUnet1D
            normalizer: MPI's LinearNormalizer for obs/action normalization
            n_obs_steps: Number of observation history steps (default 2)
            obs_feature_dim: Dimension of encoded observation features
            project_obs_embedding: Whether to project obs embeddings
            obs_embedding_projector: Optional projector for obs embeddings
        """
        super().__init__()
        self.obs_encoder = obs_encoder
        self.model = unet_model
        self.normalizer = normalizer
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.project_obs_embedding = project_obs_embedding
        self.obs_embedding_projector = obs_embedding_projector
        
    def _encode_obs(self, cond: dict) -> torch.Tensor:
        """
        Encode observations from cond dict to global conditioning vector.
        
        Args:
            cond: Dict with 'state' [B, To, Do] and 'rgb' [B, To, C, H, W]
            
        Returns:
            global_cond: [B, obs_feature_dim * n_obs_steps]
        """
        # Get batch size
        if 'rgb' in cond:
            B = cond['rgb'].shape[0]
            To = min(cond['rgb'].shape[1], self.n_obs_steps)
        else:
            B = cond['state'].shape[0]
            To = min(cond['state'].shape[1], self.n_obs_steps)
        
        # Build normalized obs dict for MPI's encoder
        # MPI expects {'images': ..., 'robot_state': ...}
        # CRITICAL: MPI normalizer transforms images from [0,1] to [-1,1]
        # The normalizer has scale=2.0, offset=-1.0 for images.
        nobs = {}
        
        # Handle RGB/images - MUST be normalized to [-1, 1]
        if 'rgb' in cond:
            rgb = cond['rgb'][:, :To]  # [B, To, C, H, W]
            # MPI's obs encoder expects [B*To, C, H, W]
            rgb_flat = rgb.reshape(-1, *rgb.shape[2:])
            # Normalize images: [0, 1] -> [-1, 1]
            normalized = self.normalizer.normalize({'images': rgb_flat})
            nobs['images'] = normalized['images']
        
        # Handle state - MPI normalizer uses 'robot_state' key
        if 'state' in cond:
            state = cond['state'][:, :To]  # [B, To, D]
            state_flat = state.reshape(-1, state.shape[-1])  # [B*To, D]
            # Normalize state using MPI's normalizer
            normalized = self.normalizer.normalize({'robot_state': state_flat})
            nobs['robot_state'] = normalized['robot_state']
        
        # Encode observations
        nobs_features = self.obs_encoder(nobs)  # [B*To, obs_feature_dim]
        
        # Project if needed
        if self.project_obs_embedding and self.obs_embedding_projector is not None:
            nobs_features = self.obs_embedding_projector(nobs_features)
        
        # Reshape to [B, To * obs_feature_dim]
        global_cond = nobs_features.reshape(B, -1)
        
        return global_cond
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: dict = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass compatible with DPPO's actor interface.
        
        Args:
            x: Noisy actions [B, Ta, action_dim]
            t: Diffusion timestep [B,] or int
            cond: Dict with 'state' and 'rgb' keys
            
        Returns:
            Predicted noise [B, Ta, action_dim]
        """
        # Encode observations to global conditioning
        global_cond = self._encode_obs(cond)
        
        # Call MPI's UNet model
        # MPI's ConditionalUnet1D: forward(sample, timestep, global_cond=...)
        noise_pred = self.model(
            sample=x,
            timestep=t,
            global_cond=global_cond,
        )
        
        return noise_pred


class MPIPPODiffusion(nn.Module):
    """
    PPO Diffusion using MPI's policy architecture.
    
    Standalone implementation that:
    1. Uses MPI's obs_encoder + UNet for the actor
    2. Supports both DDPM and DDIM sampling (controlled by use_ddim flag)
    3. Provides all methods needed for DPPO training
    
    Use the `from_mpi_checkpoints` classmethod to instantiate from config.
    """
    
    @classmethod
    def from_mpi_checkpoints(
        cls,
        # Checkpoint paths
        policy_checkpoint_path: str,
        policy_normalizer_path: str,
        critic_checkpoint_path: str = None,
        critic_normalizer_path: str = None,
        # Model params
        ft_denoising_steps: int = 10,
        denoising_steps: int = 100,
        horizon_steps: int = 32,
        obs_dim: int = 6,
        action_dim: int = 5,
        device: str = "cuda:0",
        # Critic config
        critic_n_obs_steps: int = 1,
        critic_img_cond_steps: int = 1,
        # Fresh critic option
        use_pretrained_critic: bool = True,
        critic_type: str = 'mpi',  # 'mpi' for MPI ResNet, 'vit' for DPPO ViTCritic
        critic: dict = None,  # Hydra config for fresh critic
        # PPO params
        gamma_denoising: float = 0.99,
        clip_ploss_coef: float = 0.01,
        clip_ploss_coef_base: float = 0.001,
        clip_ploss_coef_rate: float = 3,
        clip_vloss_coef: float = None,
        norm_adv: bool = True,
        # Sampling params
        min_sampling_denoising_std: float = 0.1,
        min_logprob_denoising_std: float = 0.1,
        denoised_clip_value: float = 1.0,
        randn_clip_value: float = 10,
        final_action_clip_value: float = None,
        # DDIM params (matching original DPPO)
        use_ddim: bool = False,
        ddim_steps: int = None,
        ddim_discretize: str = "uniform",
        eta = None,  # Hydra config for eta model (e.g., EtaFixed)
        learn_eta: bool = False,
        **kwargs,
    ):
        """
        Factory method to create MPIPPODiffusion from MPI checkpoint paths.
        
        This is the intended way to instantiate via hydra config.
        
        Option 1 - Pretrained MPI critic (default):
        
            model:
              _target_: model.diffusion.mpi_ppo_diffusion.MPIPPODiffusion.from_mpi_checkpoints
              policy_checkpoint_path: /path/to/policy.ckpt
              policy_normalizer_path: /path/to/normalizer.pt
              critic_checkpoint_path: /path/to/critic.pt  
              critic_normalizer_path: /path/to/critic_normalizer.pt
              use_pretrained_critic: true  # default
              ...
              
        Option 2 - Fresh MPI critic (ResNet, trained from scratch):
        
            model:
              _target_: model.diffusion.mpi_ppo_diffusion.MPIPPODiffusion.from_mpi_checkpoints
              use_pretrained_critic: false
              critic_type: mpi  # default
              critic:
                _target_: value_network.models.resnet18_tv_truck_2d_value_network.ResNet18TVTruck2dValueNetwork
                img_shape: [1, 71, 192]
                state_dim: 6
                features_dim: 512
              ...
              
        Option 3 - Fresh DPPO ViTCritic (trained from scratch):
        
            model:
              _target_: model.diffusion.mpi_ppo_diffusion.MPIPPODiffusion.from_mpi_checkpoints
              use_pretrained_critic: false
              critic_type: vit
              critic:
                _target_: model.common.critic.ViTCritic
                backbone:
                  _target_: model.common.vit.VitEncoder
                  ...
              ...
        
        Option 4 - Fresh MLP state-based critic (CriticObs, asymmetric actor-critic):
        
            model:
              _target_: model.diffusion.mpi_ppo_diffusion.MPIPPODiffusion.from_mpi_checkpoints
              use_pretrained_critic: false
              critic_type: mlp
              critic:
                _target_: model.common.critic.CriticObs
                cond_dim: 42  # full_state dimension (10 + max_boxes * 8)
                mlp_dims: [256, 256, 256]
                activation_type: Mish
                residual_style: true
              ...
              
            Note: The agent's _preprocess_obs_for_critic() extracts full_state and
            passes it as {'state': full_state} to CriticObs.
        
        Args:
            use_pretrained_critic: If True, load critic from checkpoint. If False, 
                instantiate fresh critic from `critic` config.
            critic_type: Type of fresh critic - 'mpi' for MPI ResNet, 'vit' for DPPO ViTCritic,
                'mlp' for state-based CriticObs. Only used when use_pretrained_critic=False.
            critic: Hydra config dict for fresh critic (used when use_pretrained_critic=False)
            critic_n_obs_steps: Number of state observation steps for critic (used by AGENT,
                not by the wrappers - wrappers receive pre-extracted observations)
            critic_img_cond_steps: Number of image frames for critic (used by AGENT)
        """
        from model.common.mpi_critic_wrapper import MPICriticWrapper, MPICriticWrapperFresh, DPPOViTCriticWrapper
        
        log.info("=" * 60)
        log.info("Loading MPI models for DPPO training")
        log.info("=" * 60)
        
        # Load MPI policy
        log.info(f"Loading MPI policy from {policy_checkpoint_path}")
        mpi_policy = cls._load_mpi_policy(
            policy_checkpoint_path, 
            policy_normalizer_path, 
            device
        )
        
        # Load or create critic
        if use_pretrained_critic:
            # Option 1: Load pretrained MPI value network
            if critic_checkpoint_path is None or critic_normalizer_path is None:
                raise ValueError(
                    "use_pretrained_critic=True requires critic_checkpoint_path and "
                    "critic_normalizer_path to be specified"
                )
            log.info(f"Loading pretrained MPI critic from {critic_checkpoint_path}")
            mpi_value_network = cls._load_mpi_value_network(
                critic_checkpoint_path,
                critic_normalizer_path,
                device
            )
            
            # Wrap critic for DPPO interface (uses internal normalizer)
            wrapped_critic = MPICriticWrapper(
                mpi_value_network=mpi_value_network,
                n_obs_steps=critic_n_obs_steps,
                img_cond_steps=critic_img_cond_steps,
            )
        else:
            # Option 2/3: Create fresh critic from config
            if critic is None:
                raise ValueError(
                    "use_pretrained_critic=False requires `critic` config to be specified. "
                    "Example: critic._target_: value_network.models.resnet18_tv_truck_2d_value_network.ResNet18TVTruck2dValueNetwork"
                )
            
            log.info(f"Creating fresh {critic_type.upper()} critic from config (will be trained from scratch)")
            
            # Handle both cases: critic may be a config dict OR already instantiated by Hydra
            if isinstance(critic, nn.Module):
                # Hydra already instantiated the critic (due to recursive instantiation)
                log.info(f"  Critic already instantiated: {type(critic).__name__}")
                fresh_critic = critic
            else:
                # Critic is a config dict, instantiate it
                log.info(f"  Critic config: {critic}")
                fresh_critic = hydra.utils.instantiate(critic)
            
            fresh_critic = fresh_critic.to(device)
            
            if critic_type == 'vit':
                # DPPO ViTCritic - wrap with image range conversion
                log.info("Using DPPO ViTCritic with DPPOViTCriticWrapper")
                wrapped_critic = DPPOViTCriticWrapper(fresh_critic)
                
            elif critic_type == 'mpi':
                # MPI ResNet critic - set normalizer and wrap
                log.info("Using MPI ResNet critic with MPICriticWrapperFresh")
                
                # Set the policy normalizer on the fresh MPI critic for observation normalization
                if hasattr(fresh_critic, 'set_normalizer'):
                    policy_normalizer = torch.load(
                        policy_normalizer_path, map_location=device, weights_only=False
                    )
                    fresh_critic.set_normalizer(policy_normalizer)
                    log.info("Set policy normalizer on fresh MPI critic for observation normalization")
                
                # Wrap fresh critic for DPPO interface
                wrapped_critic = MPICriticWrapperFresh(fresh_critic)
            
            elif critic_type == 'mlp':
                # MLP state-based critic (CriticObs) - no wrapper needed
                # CriticObs.forward() accepts dict with 'state' key directly
                # The agent's _preprocess_obs_for_critic() will extract full_state
                log.info("Using MLP state-based critic (CriticObs) - no wrapper")
                wrapped_critic = fresh_critic
                
            else:
                raise ValueError(f"Unknown critic_type: {critic_type}. Must be 'mpi', 'vit', or 'mlp'.")
            
            n_critic_params = sum(p.numel() for p in fresh_critic.parameters())
            n_trainable = sum(p.numel() for p in fresh_critic.parameters() if p.requires_grad)
            log.info(f"Fresh critic: {n_critic_params} total params, {n_trainable} trainable")
        
        # Create and return instance
        return cls(
            mpi_policy=mpi_policy,
            critic=wrapped_critic,
            ft_denoising_steps=ft_denoising_steps,
            denoising_steps=denoising_steps,
            horizon_steps=horizon_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            gamma_denoising=gamma_denoising,
            clip_ploss_coef=clip_ploss_coef,
            clip_ploss_coef_base=clip_ploss_coef_base,
            clip_ploss_coef_rate=clip_ploss_coef_rate,
            clip_vloss_coef=clip_vloss_coef,
            norm_adv=norm_adv,
            min_sampling_denoising_std=min_sampling_denoising_std,
            min_logprob_denoising_std=min_logprob_denoising_std,
            denoised_clip_value=denoised_clip_value,
            randn_clip_value=randn_clip_value,
            final_action_clip_value=final_action_clip_value,
            # DDIM params
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
            ddim_discretize=ddim_discretize,
            eta=eta,
            learn_eta=learn_eta,
        )
    
    @staticmethod
    def _load_mpi_policy(checkpoint_path: str, normalizer_path: str, device: str):
        """Load MPI's DiffusionUnetHybridImageTargetedPolicy from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        cfg = checkpoint.get("cfg")
        if cfg is None or not hasattr(cfg, 'policy'):
            raise ValueError("MPI checkpoint must contain 'cfg' with 'policy' config")
        
        # Instantiate policy from config
        policy = hydra.utils.instantiate(cfg.policy)
        log.info(f"Created MPI policy: {type(policy).__name__}")
        
        # Load state dict (strip DDP "module." prefix)
        state_dict = checkpoint.get("state_dicts", {}).get("model", {})
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        policy.load_state_dict(cleaned)
        log.info(f"Loaded {len(cleaned)} parameters")
        
        # Load normalizer
        normalizer = torch.load(normalizer_path, map_location=device, weights_only=False)
        policy.set_normalizer(normalizer)
        log.info(f"Set normalizer with keys: {list(normalizer.params_dict.keys())}")
        
        return policy.to(device)
    
    @staticmethod
    def _load_mpi_value_network(checkpoint_path: str, normalizer_path: str, device: str):
        """Load MPI's value network from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Try to get config, or create default
        if "cfg" in checkpoint and hasattr(checkpoint["cfg"], 'value_network'):
            value_network = hydra.utils.instantiate(checkpoint["cfg"].value_network)
        else:
            # Create default ResNet18 value network
            try:
                from value_network.models.resnet18_tv_truck_2d_value_network import ResNet18TVTruck2dValueNetwork
                value_network = ResNet18TVTruck2dValueNetwork(
                    img_shape=[1, 71, 192],
                    state_dim=6,
                    features_dim=512,
                )
                log.info("Created default ResNet18TVTruck2dValueNetwork")
            except ImportError:
                raise ImportError("Cannot import ResNet18TVTruck2dValueNetwork")
        
        # Load weights
        if "model_state_dict" in checkpoint:
            value_network.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            value_network.load_state_dict(checkpoint["state_dict"])
        elif "model" in checkpoint:
            value_network.load_state_dict(checkpoint["model"])
        else:
            value_network.load_state_dict(checkpoint)
        
        # Load normalizer
        normalizer = torch.load(normalizer_path, map_location=device, weights_only=False)
        value_network.set_normalizer(normalizer)
        
        return value_network.to(device)
    
    def __init__(
        self,
        mpi_policy,
        critic,
        ft_denoising_steps: int = 10,
        denoising_steps: int = 100,
        horizon_steps: int = 32,
        obs_dim: int = 6,
        action_dim: int = 5,
        device: str = "cuda:0",
        gamma_denoising: float = 0.99,
        clip_ploss_coef: float = 0.01,
        clip_ploss_coef_base: float = 0.001,
        clip_ploss_coef_rate: float = 3,
        clip_vloss_coef: float = None,
        norm_adv: bool = True,
        min_sampling_denoising_std: float = 0.1,
        min_logprob_denoising_std: float = 0.1,
        denoised_clip_value: float = 1.0,
        randn_clip_value: float = 10,
        final_action_clip_value: float = None,
        # DDIM params
        use_ddim: bool = False,
        ddim_steps: int = None,
        ddim_discretize: str = "uniform",
        eta = None,
        learn_eta: bool = False,
        **kwargs,
    ):
        """Initialize from already-loaded MPI policy and critic."""
        super().__init__()
        
        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.predict_epsilon = True
        
        # DDIM configuration
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps
        self.ddim_discretize = ddim_discretize
        
        # Clipping
        self.denoised_clip_value = denoised_clip_value
        self.randn_clip_value = randn_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.eps_clip_value = None  # Used only for DDIM epsilon clipping
        
        # Create actor wrapper
        self.actor = MPIPolicyActorWrapper(
            obs_encoder=mpi_policy.obs_encoder,
            unet_model=mpi_policy.model,
            normalizer=mpi_policy.normalizer,
            n_obs_steps=mpi_policy.n_obs_steps,
            obs_feature_dim=mpi_policy.obs_feature_dim,
            project_obs_embedding=mpi_policy.project_obs_embedding,
            obs_embedding_projector=getattr(mpi_policy, 'obs_embedding_projector', None),
        ).to(device)
        self.network = self.actor
        
        # Finetuned copy
        self.actor_ft = copy.deepcopy(self.actor)
        for param in self.actor.parameters():
            param.requires_grad = False
        
        n_params = sum(p.numel() for p in self.actor_ft.parameters() if p.requires_grad)
        log.info(f"Actor: {n_params} trainable parameters")
        
        # Critic
        self.critic = critic.to(device)
        self.mpi_normalizer = mpi_policy.normalizer
        
        # Store MPI's noise scheduler for inference (uses exact same sampling as MPI)
        self.mpi_noise_scheduler = mpi_policy.noise_scheduler
        self.use_mpi_scheduler = True  # Use MPI scheduler for sampling
        log.info(f"Using MPI noise scheduler: {type(self.mpi_noise_scheduler).__name__}")
        
        # Fine-tuning config
        self.ft_denoising_steps = ft_denoising_steps
        self.ft_denoising_steps_d = 0
        self.ft_denoising_steps_t = 0
        self.ft_denoising_steps_cnt = 0
        
        # Sampling
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std
        
        # PPO
        self.gamma_denoising = gamma_denoising
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.clip_vloss_coef = clip_vloss_coef
        self.norm_adv = norm_adv
        self.clip_advantage_lower_quantile = 0
        self.clip_advantage_upper_quantile = 1
        
        # Validate DDIM configuration
        assert ft_denoising_steps <= denoising_steps, \
            f"ft_denoising_steps ({ft_denoising_steps}) must be <= denoising_steps ({denoising_steps})"
        if use_ddim:
            assert ddim_steps is not None, "ddim_steps must be specified when use_ddim=True"
            assert ft_denoising_steps <= ddim_steps, \
                f"ft_denoising_steps ({ft_denoising_steps}) must be <= ddim_steps ({ddim_steps})"
            assert not (learn_eta and not use_ddim), "Cannot learn eta with DDPM"
        
        # Eta model for DDIM variance control
        # For DDPM, eta is not used (variance is fixed by the schedule)
        self.learn_eta = learn_eta
        if use_ddim and eta is not None:
            # Handle eta: may be already instantiated or a config dict
            if isinstance(eta, nn.Module):
                self.eta = eta.to(self.device)
            else:
                # Instantiate from hydra config
                self.eta = hydra.utils.instantiate(eta).to(self.device)
            if not learn_eta:
                for param in self.eta.parameters():
                    param.requires_grad = False
                log.info("Turned off gradients for eta (not learning)")
            log.info("Initialized eta model: %s", type(self.eta).__name__)
        else:
            self.eta = None
        
        # DDPM parameters - always needed for log prob computation
        self._setup_ddpm_params(denoising_steps)
        
        # DDIM parameters - only setup if using DDIM
        if use_ddim:
            self._setup_ddim_params(denoising_steps, ddim_steps, ddim_discretize)
            # Disable MPI scheduler for DDIM (use our implementation)
            self.use_mpi_scheduler = False
            log.info("MPIPPODiffusion: DDIM with %d steps, finetuning last %d", ddim_steps, ft_denoising_steps)
        else:
            log.info("MPIPPODiffusion: DDPM with %d steps, finetuning last %d", denoising_steps, ft_denoising_steps)
            log.info("  use_mpi_scheduler=%s", self.use_mpi_scheduler)
    
    def _setup_ddpm_params(self, denoising_steps: int):
        """
        Set up DDPM noise schedule parameters.
        
        If MPI scheduler is available, extracts alphas/betas from it to ensure
        exact consistency between sampling (using scheduler) and training 
        (using these parameters for log prob computation).
        """
        # Try to use MPI scheduler's parameters for consistency
        if hasattr(self, 'mpi_noise_scheduler') and self.mpi_noise_scheduler is not None:
            scheduler = self.mpi_noise_scheduler
            # Extract alphas_cumprod from scheduler
            self.alphas_cumprod = scheduler.alphas_cumprod.to(self.device)
            self.alphas = scheduler.alphas.to(self.device)
            self.betas = scheduler.betas.to(self.device)
            log.info("Using DDPM parameters from MPI noise scheduler (squaredcos_cap_v2)")
        else:
            # Fallback to our cosine schedule
            self.betas = cosine_beta_schedule(denoising_steps).to(self.device)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
            log.info("Using custom cosine beta schedule")
        
        # α̅_{t-1}
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), self.alphas_cumprod[:-1]]
        )
        
        # √α̅_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        
        # √(1 - α̅_t)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # √(1/α̅_t)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        
        # √(1/α̅_t - 1)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # β̃_t = σ_t² = β_t * (1 - α̅_{t-1}) / (1 - α̅_t)
        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))
        
        # μ_t coefficients for x_0 reconstruction
        self.ddpm_mu_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_mu_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _setup_ddim_params(self, denoising_steps: int, ddim_steps: int, ddim_discretize: str):
        """
        Set up DDIM-specific parameters for accelerated sampling.
        
        DDIM (Denoising Diffusion Implicit Models) allows for faster sampling by
        using a non-Markovian process. This method sets up the timestep schedule
        and alpha values needed for DDIM sampling.
        
        Following the original DPPO implementation in diffusion.py lines 155-196.
        
        Args:
            denoising_steps: Total number of DDPM denoising steps (e.g., 100)
            ddim_steps: Number of DDIM steps to use (e.g., 5)
            ddim_discretize: How to select DDIM timesteps ("uniform" supported)
        """
        assert self.predict_epsilon, "DDIM requires predicting epsilon"
        
        # Create DDIM timestep schedule
        # uniform: evenly spaced timesteps from the full schedule
        if ddim_discretize == "uniform":
            step_ratio = denoising_steps // ddim_steps
            # ddim_t contains the actual timestep values (e.g., [0, 20, 40, 60, 80] for 5 steps)
            self.ddim_t = (
                torch.arange(0, ddim_steps, device=self.device) * step_ratio
            )
        else:
            raise ValueError(f"Unknown DDIM discretization method: {ddim_discretize}")
        
        # Extract alpha values at DDIM timesteps
        # α̅_t at DDIM timesteps
        self.ddim_alphas = (
            self.alphas_cumprod[self.ddim_t].clone().to(torch.float32)
        )
        # √α̅_t
        self.ddim_alphas_sqrt = torch.sqrt(self.ddim_alphas)
        # α̅_{t-1} (previous step's cumulative alpha)
        self.ddim_alphas_prev = torch.cat(
            [
                torch.tensor([1.0]).to(torch.float32).to(self.device),
                self.alphas_cumprod[self.ddim_t[:-1]],
            ]
        )
        # √(1-α̅_t)
        self.ddim_sqrt_one_minus_alphas = (1.0 - self.ddim_alphas) ** 0.5
        
        # Initialize fixed sigmas for inference with eta=0 (deterministic DDIM)
        # σ_t = η * √((1-α̅_{t-1})/(1-α̅_t)) * √(1 - α̅_t/α̅_{t-1})
        ddim_eta = 0  # Deterministic by default, can be overridden by eta model
        self.ddim_sigmas = (
            ddim_eta
            * (
                (1 - self.ddim_alphas_prev)
                / (1 - self.ddim_alphas)
                * (1 - self.ddim_alphas / self.ddim_alphas_prev)
            )
            ** 0.5
        )
        
        # Flip all arrays for reverse diffusion (sampling goes from T to 0)
        # After flip: indices go high-to-low timestep
        self.ddim_t = torch.flip(self.ddim_t, [0])
        self.ddim_alphas = torch.flip(self.ddim_alphas, [0])
        self.ddim_alphas_sqrt = torch.flip(self.ddim_alphas_sqrt, [0])
        self.ddim_alphas_prev = torch.flip(self.ddim_alphas_prev, [0])
        self.ddim_sqrt_one_minus_alphas = torch.flip(self.ddim_sqrt_one_minus_alphas, [0])
        self.ddim_sigmas = torch.flip(self.ddim_sigmas, [0])
        
        log.info("DDIM schedule: %d steps, timesteps=%s", ddim_steps, self.ddim_t.tolist())
    
    def p_mean_var(
        self,
        x,
        t,
        cond,
        index=None,
        use_base_policy=False,
        deterministic=False,
    ):
        """
        Compute mean and variance for denoising step.
        
        Supports both DDPM and DDIM sampling, following the original DPPO pattern:
        1. Always use frozen base actor first for ALL samples
        2. Then overwrite only ft_indices with actor_ft predictions
        
        For DDPM:
        - FT steps: t < ft_denoising_steps
        - Uses DDPM posterior variance formula
        
        For DDIM:
        - FT steps: index >= (ddim_steps - ft_denoising_steps)
        - Uses learned eta model for variance control
        
        Args:
            x: Current noisy sample [B, Ta, Da]
            t: Timestep [B,] (actual diffusion timestep values)
            cond: Conditioning dict {'state': [B,To,Do], 'rgb': [B,To,C,H,W]}
            index: DDIM step index [B,] (0 to ddim_steps-1), required for DDIM
            use_base_policy: Whether to use frozen base policy for ALL steps
            deterministic: Whether to use deterministic sampling (eta=0 for DDIM)
            
        Returns:
            mu: Mean [B, Ta, Da]
            logvar: Log variance [B, Ta, Da]
            etas: Eta values [B, 1, 1] or [B, 1, Da] for DDIM, ones for DDPM
        """
        # ALWAYS use frozen base actor first for ALL samples
        # This matches original DPPO behavior
        noise = self.actor(x, t, cond=cond)
        
        # Determine which samples are in fine-tuning range
        # The logic differs between DDPM and DDIM
        if self.use_ddim:
            # For DDIM: finetune the last ft_denoising_steps based on index
            # index goes from 0 (high timestep) to ddim_steps-1 (low timestep)
            # FT range: index >= (ddim_steps - ft_denoising_steps)
            ft_indices = torch.where(
                index >= (self.ddim_steps - self.ft_denoising_steps)
            )[0]
        else:
            # For DDPM: finetune steps where t < ft_denoising_steps
            ft_indices = torch.where(t < self.ft_denoising_steps)[0]
        
        # Select which actor to use for finetuning steps
        # use_base_policy=True means use base for everything (e.g., for BC loss)
        actor_for_ft = self.actor if use_base_policy else self.actor_ft
        
        # Overwrite ft_indices with finetuned (or base if use_base_policy) predictions
        if len(ft_indices) > 0:
            cond_ft = {key: cond[key][ft_indices] for key in cond}
            noise_ft = actor_for_ft(x[ft_indices], t[ft_indices], cond=cond_ft)
            noise[ft_indices] = noise_ft
        
        # Predict x_0 from noise (epsilon prediction)
        if self.predict_epsilon:
            if self.use_ddim:
                # DDIM x_0 reconstruction: x₀ = (xₜ - √(1-αₜ) * ε) / √αₜ
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(
                    self.ddim_sqrt_one_minus_alphas, index, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
            else:
                # DDPM x_0 reconstruction: x₀ = √(1/α̅ₜ) * xₜ - √(1/α̅ₜ - 1) * ε
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:
            # Directly predicting x_0
            x_recon = noise
        
        # Clip reconstructed sample
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # Re-calculate noise based on clamped x_recon for DDIM
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha
        
        # Clip epsilon for numerical stability in policy gradient (DDIM only)
        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)
        
        # Compute mean and variance based on sampling method
        if self.use_ddim:
            # DDIM: μ = √α_{t-1} * x₀ + √(1-α_{t-1} - σ²) * ε
            if deterministic:
                # Deterministic DDIM: eta = 0
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                # Use eta model for stochastic DDIM
                etas = self.eta(cond).unsqueeze(1)  # B x 1 x (Da or 1)
            
            # Compute sigma using eta
            # σ = η * √((1-α_{t-1})/(1-αₜ) * (1 - αₜ/α_{t-1}))
            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)) ** 0.5
            ).clamp_(min=1e-10)
            
            # Direction pointing to x_t coefficient
            dir_xt_coef = (1.0 - alpha_prev - sigma**2).clamp_(min=0).sqrt()
            
            # Mean: μ = √α_{t-1} * x₀ + dir_xt_coef * ε
            mu = (alpha_prev**0.5) * x_recon + dir_xt_coef * noise
            
            # Variance
            var = sigma**2
            logvar = torch.log(var)
        else:
            # DDPM: μₜ = β̃ₜ * √α̅_{t-1}/(1-α̅ₜ) * x₀ + √αₜ * (1-α̅_{t-1})/(1-α̅ₜ) * xₜ
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
            # For DDPM, eta is always 1 (fixed variance)
            etas = torch.ones_like(mu).to(mu.device)
        
        return mu, logvar, etas
    
    def _sample_with_mpi_scheduler(
        self,
        cond,
        return_chain=True,
        use_base_policy=False,
    ):
        """
        Sample actions using MPI's diffusers-based noise scheduler.
        This produces identical results to MPI's conditional_sample method.
        
        Following original DPPO pattern:
        - For t >= ft_denoising_steps: use frozen base actor (base obs_encoder + base UNet)
        - For t < ft_denoising_steps: use finetuned actor (ft obs_encoder + ft UNet)
        
        Args:
            cond: Dict with 'state' and 'rgb' keys
            return_chain: Whether to return denoising chain
            use_base_policy: Whether to use frozen base policy for ALL steps
            
        Returns:
            Sample namedtuple with:
                trajectories: Final actions [B, Ta, Da]
                chains: Denoising chain [B, K+1, Ta, Da] if return_chain else None
        """
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)
        
        # Use MPI's scheduler
        scheduler = self.mpi_noise_scheduler
        scheduler.set_timesteps(self.denoising_steps)
        
        # Encode observations with BOTH actors upfront (for efficiency)
        # We'll use the appropriate encoding at each timestep
        global_cond_base = self.actor._encode_obs(cond)
        
        # For ft steps, use ft actor's encoding (unless use_base_policy=True)
        if use_base_policy:
            global_cond_ft = global_cond_base  # Use base for everything
        else:
            global_cond_ft = self.actor_ft._encode_obs(cond)
        
        # Start from pure noise
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        
        # Collect chain if requested
        chain = [] if return_chain else None
        
        # Add initial noise to chain if finetuning all steps
        if return_chain and self.ft_denoising_steps == self.denoising_steps:
            chain.append(x.clone())
        
        # DDPM sampling using scheduler.step()
        # Switch between base and ft actors based on timestep
        for t in scheduler.timesteps:
            t_tensor = t.unsqueeze(0).expand(B).to(device)
            t_val = t.item()
            
            # Determine which actor/encoding to use based on timestep
            # t >= ft_denoising_steps: use frozen base actor
            # t < ft_denoising_steps: use finetuned actor (unless use_base_policy)
            if t_val >= self.ft_denoising_steps:
                # Non-finetuning range: use frozen base actor
                model = self.actor.model
                global_cond = global_cond_base
            else:
                # Finetuning range: use ft actor (or base if use_base_policy)
                model = self.actor.model if use_base_policy else self.actor_ft.model
                global_cond = global_cond_base if use_base_policy else global_cond_ft
            
            # Predict noise (epsilon)
            noise_pred = model(
                sample=x,
                timestep=t_tensor,
                global_cond=global_cond,
            )
            
            # Scheduler step: x_t -> x_{t-1}
            x = scheduler.step(noise_pred, t, x).prev_sample
            
            # Add to chain for finetuning steps
            if return_chain and t_val <= self.ft_denoising_steps:
                chain.append(x.clone())
        
        if return_chain:
            chain = torch.stack(chain, dim=1)
        
        return Sample(x, chain)
    
    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        return_chain=True,
        use_base_policy=False,
    ):
        """
        Sample actions using DDPM or DDIM depending on configuration.
        
        For DDPM with use_mpi_scheduler=True: uses the diffusers-based scheduler
        for exact compatibility with MPI's sampling.
        
        For DDIM or DDPM with use_mpi_scheduler=False: uses our custom implementation
        that supports fine-tuning with policy gradient.
        
        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            deterministic: If true, then std=0 with DDIM, or with DDPM, use normal
                schedule (instead of clipping at a higher value)
            return_chain: whether to return the entire chain of denoised actions
            use_base_policy: whether to use the frozen pre-trained policy instead
            
        Returns:
            Sample namedtuple with:
                trajectories: Final actions [B, Ta, Da]
                chains: Denoising chain [B, K+1, Ta, Da] if return_chain else None
        """
        # Use MPI scheduler for DDPM inference if available and enabled
        # (DDIM uses our custom implementation to support eta model)
        if not self.use_ddim and self.use_mpi_scheduler and hasattr(self, 'mpi_noise_scheduler'):
            return self._sample_with_mpi_scheduler(
                cond=cond,
                return_chain=return_chain,
                use_base_policy=use_base_policy,
            )
        
        # Custom DDPM/DDIM implementation
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)
        
        # Get minimum sampling std
        min_sampling_denoising_std = self.get_min_sampling_denoising_std()
        
        # Start from pure noise
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        
        # Timestep schedule depends on DDPM vs DDIM
        if self.use_ddim:
            t_all = self.ddim_t  # Already in reverse order (high to low timestep)
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        
        # Collect chain if requested
        chain = [] if return_chain else None
        
        # Add initial noise to chain if finetuning all steps
        if not self.use_ddim and self.ft_denoising_steps == self.denoising_steps:
            if return_chain:
                chain.append(x)
        if self.use_ddim and self.ft_denoising_steps == self.ddim_steps:
            if return_chain:
                chain.append(x)
        
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            
            mean, logvar, _ = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
                use_base_policy=use_base_policy,
                deterministic=deterministic,
            )
            std = torch.exp(0.5 * logvar)
            
            # Determine noise level
            if self.use_ddim:
                # DDIM noise handling
                if deterministic:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=min_sampling_denoising_std)
            else:
                # DDPM noise handling
                if deterministic and t == 0:
                    std = torch.zeros_like(std)
                elif deterministic:
                    std = torch.clip(std, min=1e-3)
                else:
                    std = torch.clip(std, min=min_sampling_denoising_std)
            
            # Sample
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise
            
            # Clamp at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )
            
            # Add to chain for finetuning steps
            if return_chain:
                if not self.use_ddim and t <= self.ft_denoising_steps:
                    chain.append(x)
                elif self.use_ddim and i >= (self.ddim_steps - self.ft_denoising_steps - 1):
                    chain.append(x)
        
        if return_chain:
            chain = torch.stack(chain, dim=1)
        
        return Sample(x, chain)
    
    def step(self):
        """Anneal min_sampling_denoising_std and fine-tuning denoising steps."""
        # Anneal min_sampling_denoising_std
        if type(self.min_sampling_denoising_std) is not float:
            self.min_sampling_denoising_std.step()
        
        # Anneal denoising steps (if configured)
        self.ft_denoising_steps_cnt += 1
        if (
            self.ft_denoising_steps_d > 0
            and self.ft_denoising_steps_t > 0
            and self.ft_denoising_steps_cnt % self.ft_denoising_steps_t == 0
        ):
            self.ft_denoising_steps = max(
                0, self.ft_denoising_steps - self.ft_denoising_steps_d
            )
            # Update actor
            self.actor = self.actor_ft
            self.actor_ft = copy.deepcopy(self.actor)
            for param in self.actor.parameters():
                param.requires_grad = False
            log.info(
                f"Finished annealing fine-tuning denoising steps to {self.ft_denoising_steps}"
            )
    
    def get_min_sampling_denoising_std(self):
        """Get current minimum sampling denoising std."""
        if type(self.min_sampling_denoising_std) is float:
            return self.min_sampling_denoising_std
        else:
            return self.min_sampling_denoising_std()
    
    # ========== RL Training Methods ==========
    
    def get_logprobs(
        self,
        cond,
        chains,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """
        Compute log probabilities of the entire chain of denoised actions.
        
        Supports both DDPM and DDIM. For DDIM, uses ddim_t timesteps and
        index-based addressing into the DDIM schedule.
        
        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da) where K = ft_denoising_steps
            get_ent: flag for returning entropy (actually returns eta)
            use_base_policy: flag for using base policy
            
        Returns:
            logprobs: (B x K, Ta, Da)
            eta (if get_ent=True): (B x K, Ta)
        """
        from torch.distributions import Normal
        
        # Repeat cond for denoising steps, flatten batch and time dimensions
        cond_expanded = {
            key: cond[key]
            .unsqueeze(1)
            .repeat(1, self.ft_denoising_steps, *(1,) * (cond[key].ndim - 1))
            .flatten(start_dim=0, end_dim=1)
            for key in cond
        }
        
        # Get timesteps based on DDPM or DDIM
        if self.use_ddim:
            # DDIM: use the last ft_denoising_steps from ddim_t
            t_single = self.ddim_t[-self.ft_denoising_steps:]
        else:
            # DDPM: timesteps from ft_denoising_steps-1 down to 0
            t_single = torch.arange(
                start=self.ft_denoising_steps - 1,
                end=-1,
                step=-1,
                device=self.device,
            )
        # Repeat for each batch item: [4,3,2,1,0, 4,3,2,1,0, ...]
        t_all = t_single.repeat(chains.shape[0], 1).flatten()
        
        # For DDIM, also need indices into the DDIM schedule
        if self.use_ddim:
            indices_single = torch.arange(
                start=self.ddim_steps - self.ft_denoising_steps,
                end=self.ddim_steps,
                device=self.device,
            )
            indices = indices_single.repeat(chains.shape[0])
        else:
            indices = None
        
        # Split chains into prev/next pairs
        chains_prev = chains[:, :-1]
        chains_next = chains[:, 1:]
        
        # Flatten first two dimensions: (B, K, Ta, Da) -> (B*K, Ta, Da)
        chains_prev = chains_prev.reshape(-1, self.horizon_steps, self.action_dim)
        chains_next = chains_next.reshape(-1, self.horizon_steps, self.action_dim)
        
        # Get mean/var from forward pass
        mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond_expanded,
            index=indices,
            use_base_policy=use_base_policy,
        )
        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        
        # Compute log prob with Gaussian
        dist = Normal(mean, std)
        log_prob = dist.log_prob(chains_next)
        
        if get_ent:
            return log_prob, eta
        return log_prob
    
    def get_logprobs_subsample(
        self,
        cond,
        chains_prev,
        chains_next,
        denoising_inds,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """
        Compute log probabilities for random samples of denoised chains.
        
        This is used during PPO training where we subsample denoising steps
        for efficiency rather than computing logprobs for all steps.
        
        Supports both DDPM and DDIM. For DDIM, uses ddim_t timesteps and
        index-based addressing into the DDIM schedule.
        
        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains_prev: Previous chain states [B, Ta, Da]
            chains_next: Next chain states [B, Ta, Da]
            denoising_inds: Indices of denoising steps [B,] (0 to ft_denoising_steps-1)
            get_ent: flag for returning entropy (actually returns eta)
            use_base_policy: flag for using base policy
            
        Returns:
            logprobs: (B, Ta, Da)
            eta (if get_ent=True): (B, Ta)
        """
        from torch.distributions import Normal
        
        # Get timesteps for these indices based on DDPM or DDIM
        if self.use_ddim:
            # DDIM: use the last ft_denoising_steps from ddim_t
            t_single = self.ddim_t[-self.ft_denoising_steps:]
        else:
            # DDPM: timesteps from ft_denoising_steps-1 down to 0
            t_single = torch.arange(
                start=self.ft_denoising_steps - 1,
                end=-1,
                step=-1,
                device=self.device,
            )
        # Select timesteps for the subsampled indices
        t_all = t_single[denoising_inds]
        
        # For DDIM, also need indices into the DDIM schedule
        if self.use_ddim:
            ddim_indices_single = torch.arange(
                start=self.ddim_steps - self.ft_denoising_steps,
                end=self.ddim_steps,
                device=self.device,
            )
            ddim_indices = ddim_indices_single[denoising_inds]
        else:
            ddim_indices = None
        
        # Get mean/var from forward pass
        mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=ddim_indices,
            use_base_policy=use_base_policy,
        )
        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        
        # Compute log prob with Gaussian
        dist = Normal(mean, std)
        log_prob = dist.log_prob(chains_next)
        
        if get_ent:
            return log_prob, eta
        return log_prob
    
    def loss(
        self,
        obs,
        chains_prev,
        chains_next,
        denoising_inds,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        reward_horizon=4,
        critic_obs=None,
    ):
        """
        PPO loss computation.
        
        Args:
            obs: Dict with 'state' and 'rgb'
            chains_prev/next: Denoising chain states
            denoising_inds: Indices into denoising steps
            returns: Target returns
            oldvalues: Old value estimates
            advantages: Advantage estimates
            oldlogprobs: Old log probabilities
            use_bc_loss: Whether to add BC loss (not implemented for MPI)
            reward_horizon: Action horizon for gradient
            critic_obs: Optional separate obs for critic
            
        Returns:
            Tuple of losses and metrics
        """
        import math
        
        # Get new logprobs
        newlogprobs, eta = self.get_logprobs_subsample(
            obs, chains_prev, chains_next, denoising_inds, get_ent=True
        )
        entropy_loss = -eta.mean()
        newlogprobs = newlogprobs.clamp(min=-5, max=2)
        oldlogprobs = oldlogprobs.clamp(min=-5, max=2)
        
        # Only backprop through reward_horizon steps
        newlogprobs = newlogprobs[:, :reward_horizon, :]
        oldlogprobs = oldlogprobs[:, :reward_horizon, :]
        
        # Average over action dims
        newlogprobs = newlogprobs.mean(dim=(-1, -2)).view(-1)
        oldlogprobs = oldlogprobs.mean(dim=(-1, -2)).view(-1)
        
        # BC loss (not implemented for MPI)
        bc_loss = 0
        
        # Normalize advantages
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Clip advantages
        adv_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=adv_min, max=adv_max)
        
        # Denoising discount
        discount = torch.tensor([
            self.gamma_denoising ** (self.ft_denoising_steps - i - 1)
            for i in denoising_inds
        ]).to(self.device)
        advantages = advantages * discount
        
        # Ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()
        
        # Interpolated clipping
        t = (denoising_inds.float() / (self.ft_denoising_steps - 1)).to(self.device)
        if self.ft_denoising_steps > 1:
            clip_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (torch.exp(self.clip_ploss_coef_rate * t) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1
            )
        else:
            clip_coef = t
        
        # KL and clip fraction
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()
        
        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        # Value loss
        critic_input = critic_obs if critic_obs is not None else obs
        newvalues = self.critic(critic_input).view(-1)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch.clamp(
                newvalues - oldvalues, -self.clip_vloss_coef, self.clip_vloss_coef
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            bc_loss,
            eta.mean().item(),
        )
