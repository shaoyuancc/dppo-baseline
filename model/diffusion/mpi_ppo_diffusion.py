"""
MPI-DPPO Integration: PPO Diffusion with MPI's policy architecture.

This module provides wrapper classes to adapt MPI's DiffusionUnetHybridImageTargetedPolicy
to DPPO's training interface, enabling PPO-based finetuning of MPI checkpoints.

Key components:
- MPIPolicyActorWrapper: Wraps MPI's obs_encoder + UNet for DPPO-compatible interface
- MPIPPODiffusion: DPPO model using MPI's architecture with DDPM sampling

Usage in config:
    model:
      _target_: model.diffusion.mpi_ppo_diffusion.MPIPPODiffusion.from_mpi_checkpoints
      policy_checkpoint_path: /path/to/policy.ckpt
      policy_normalizer_path: /path/to/normalizer.pt
      critic_checkpoint_path: /path/to/critic.pt
      critic_normalizer_path: /path/to/critic_normalizer.pt
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
    2. Uses DDPM for denoising (not DDIM)
    3. Provides all methods needed for DPPO training
    
    Use the `from_mpi_checkpoints` classmethod to instantiate from config.
    """
    
    @classmethod
    def from_mpi_checkpoints(
        cls,
        # Checkpoint paths
        policy_checkpoint_path: str,
        policy_normalizer_path: str,
        critic_checkpoint_path: str,
        critic_normalizer_path: str,
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
        **kwargs,
    ):
        """
        Factory method to create MPIPPODiffusion from MPI checkpoint paths.
        
        This is the intended way to instantiate via hydra config:
        
            model:
              _target_: model.diffusion.mpi_ppo_diffusion.MPIPPODiffusion.from_mpi_checkpoints
              policy_checkpoint_path: /path/to/policy.ckpt
              policy_normalizer_path: /path/to/normalizer.pt
              critic_checkpoint_path: /path/to/critic.pt  
              critic_normalizer_path: /path/to/critic_normalizer.pt
              ft_denoising_steps: 10
              ...
        """
        from model.common.mpi_critic_wrapper import MPICriticWrapper
        
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
        
        # Load MPI value network  
        log.info(f"Loading MPI critic from {critic_checkpoint_path}")
        mpi_value_network = cls._load_mpi_value_network(
            critic_checkpoint_path,
            critic_normalizer_path,
            device
        )
        
        # Wrap critic for DPPO interface
        critic = MPICriticWrapper(
            mpi_value_network=mpi_value_network,
            n_obs_steps=critic_n_obs_steps,
            img_cond_steps=critic_img_cond_steps,
        )
        
        # Create and return instance
        return cls(
            mpi_policy=mpi_policy,
            critic=critic,
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
        self.use_ddim = False
        self.ddim_steps = None
        
        # Clipping
        self.denoised_clip_value = denoised_clip_value
        self.randn_clip_value = randn_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.eps_clip_value = None
        
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
        
        # No eta for DDPM
        self.learn_eta = False
        self.eta = None
        
        # DDPM parameters - sync with MPI scheduler if available
        self._setup_ddpm_params(denoising_steps)
        
        log.info(f"MPIPPODiffusion: {denoising_steps} steps, finetuning last {ft_denoising_steps}")
        log.info(f"  use_mpi_scheduler={self.use_mpi_scheduler}")
    
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
        
        Overrides VPGDiffusion.p_mean_var to use MPI's architecture.
        
        Args:
            x: Current noisy sample [B, Ta, Da]
            t: Timestep [B,]
            cond: Conditioning dict {'state': [B,To,Do], 'rgb': [B,To,C,H,W]}
            index: Not used for DDPM
            use_base_policy: Whether to use frozen base policy
            deterministic: Whether to use deterministic sampling
            
        Returns:
            mu: Mean [B, Ta, Da]
            logvar: Log variance [B, Ta, Da]
            etas: Ones for DDPM (no eta learning)
        """
        # Select actor (base or finetuned)
        actor = self.actor if use_base_policy else self.actor_ft
        
        # Get noise prediction
        noise = actor(x, t, cond=cond)
        
        # Determine which samples are in fine-tuning range
        # For DDPM: finetune steps where t < ft_denoising_steps
        ft_indices = torch.where(t < self.ft_denoising_steps)[0]
        
        # If not using base policy and some samples are in ft range,
        # use finetuned actor for those
        if not use_base_policy and len(ft_indices) > 0 and len(ft_indices) < len(t):
            # Get predictions from base actor for non-ft samples
            base_noise = self.actor(x, t, cond=cond)
            # Overwrite ft samples with finetuned predictions
            noise = base_noise.clone()
            if len(ft_indices) > 0:
                cond_ft = {key: cond[key][ft_indices] for key in cond}
                noise_ft = self.actor_ft(x[ft_indices], t[ft_indices], cond=cond_ft)
                noise[ft_indices] = noise_ft
        
        # Predict x_0 from noise (epsilon prediction)
        # x_0 = √(1/α̅_t) * x_t - √(1/α̅_t - 1) * ε
        x_recon = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
        )
        
        # Clip reconstructed sample
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
        
        # Compute posterior mean
        # μ_t = β̃_t * √α̅_{t-1} / (1-α̅_t) * x_0 + √α_t * (1-α̅_{t-1}) / (1-α̅_t) * x_t
        mu = (
            extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
            + extract(self.ddpm_mu_coef2, t, x.shape) * x
        )
        
        # Get log variance
        logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
        
        # Return ones for eta (DDPM doesn't use learned eta)
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
        
        Args:
            cond: Dict with 'state' and 'rgb' keys
            return_chain: Whether to return denoising chain
            use_base_policy: Whether to use frozen base policy
            
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
        
        # Select actor
        actor = self.actor if use_base_policy else self.actor_ft
        
        # Encode observations once (for efficiency)
        global_cond = actor._encode_obs(cond)
        
        # Start from pure noise
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        
        # Collect chain if requested
        chain = [] if return_chain else None
        
        # Add initial noise to chain if finetuning all steps
        if return_chain and self.ft_denoising_steps == self.denoising_steps:
            chain.append(x.clone())
        
        # DDPM sampling using scheduler.step() - exactly like MPI
        for t in scheduler.timesteps:
            t_tensor = t.unsqueeze(0).expand(B).to(device)
            
            # Predict noise (epsilon)
            noise_pred = actor.model(
                sample=x,
                timestep=t_tensor,
                global_cond=global_cond,
            )
            
            # Scheduler step: x_t -> x_{t-1}
            x = scheduler.step(noise_pred, t, x).prev_sample
            
            # Add to chain for finetuning steps
            if return_chain and t.item() <= self.ft_denoising_steps:
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
        Sample actions using DDPM.
        
        If use_mpi_scheduler is True, uses the diffusers-based scheduler
        for exact compatibility with MPI's sampling. Otherwise uses custom
        DDPM implementation.
        
        Args:
            cond: Dict with 'state' and 'rgb' keys
            deterministic: If True, use zero noise at t=0 (only for custom sampling)
            return_chain: Whether to return denoising chain
            use_base_policy: Whether to use frozen base policy
            
        Returns:
            Sample namedtuple with:
                trajectories: Final actions [B, Ta, Da]
                chains: Denoising chain [B, K+1, Ta, Da] if return_chain else None
        """
        # Use MPI scheduler for inference if available and enabled
        if self.use_mpi_scheduler and hasattr(self, 'mpi_noise_scheduler'):
            return self._sample_with_mpi_scheduler(
                cond=cond,
                return_chain=return_chain,
                use_base_policy=use_base_policy,
            )
        
        # Fallback to custom DDPM implementation
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)
        
        # Get minimum sampling std
        min_sampling_denoising_std = self.get_min_sampling_denoising_std()
        
        # Start from pure noise
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        
        # DDPM timesteps (T-1 to 0)
        t_all = list(reversed(range(self.denoising_steps)))
        
        # Collect chain if requested
        chain = [] if return_chain else None
        
        # Add initial noise to chain if finetuning all steps
        if return_chain and self.ft_denoising_steps == self.denoising_steps:
            chain.append(x.clone())
        
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            
            mean, logvar, _ = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                use_base_policy=use_base_policy,
                deterministic=deterministic,
            )
            std = torch.exp(0.5 * logvar)
            
            # Determine noise level
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
            if return_chain and t <= self.ft_denoising_steps:
                chain.append(x.clone())
        
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
        Compute log probabilities of denoised action chains.
        
        Args:
            cond: Dict with 'state' and 'rgb' keys
            chains: Denoising chain [B, K+1, Ta, Da]
            get_ent: Whether to return entropy
            use_base_policy: Whether to use base policy
            
        Returns:
            logprobs: [B*K, Ta, Da]
            entropy: [B*K, Ta] if get_ent else None
        """
        from torch.distributions import Normal
        
        # Repeat cond for denoising steps
        cond_expanded = {
            key: cond[key]
            .unsqueeze(1)
            .repeat(1, self.ft_denoising_steps, *(1,) * (cond[key].ndim - 1))
            .flatten(start_dim=0, end_dim=1)
            for key in cond
        }
        
        # DDPM timesteps for fine-tuning range
        t_single = torch.arange(
            start=self.ft_denoising_steps - 1,
            end=-1,
            step=-1,
            device=self.device,
        )
        t_all = t_single.repeat(chains.shape[0], 1).flatten()
        
        # Split chains into prev/next
        chains_prev = chains[:, :-1].reshape(-1, self.horizon_steps, self.action_dim)
        chains_next = chains[:, 1:].reshape(-1, self.horizon_steps, self.action_dim)
        
        # Get mean/var
        mean, logvar, eta = self.p_mean_var(
            chains_prev, t_all, cond=cond_expanded, use_base_policy=use_base_policy
        )
        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        
        # Compute log prob
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
        Compute log probabilities for subsampled denoising steps.
        
        Args:
            cond: Dict with 'state' and 'rgb'
            chains_prev: Previous chain states [B, Ta, Da]
            chains_next: Next chain states [B, Ta, Da]
            denoising_inds: Indices of denoising steps [B,]
            get_ent: Whether to return entropy
            use_base_policy: Whether to use base policy
            
        Returns:
            logprobs: [B, Ta, Da]
            eta: [B, Ta] if get_ent
        """
        from torch.distributions import Normal
        
        # Get timesteps for these indices
        t_single = torch.arange(
            start=self.ft_denoising_steps - 1,
            end=-1,
            step=-1,
            device=self.device,
        )
        t_all = t_single[denoising_inds]
        
        # Get mean/var
        mean, logvar, eta = self.p_mean_var(
            chains_prev, t_all, cond=cond, use_base_policy=use_base_policy
        )
        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        
        # Compute log prob
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
