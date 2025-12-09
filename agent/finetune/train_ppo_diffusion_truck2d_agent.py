"""
DPPO fine-tuning for truck_2d with custom metrics.

Extends TrainPPOImgDiffusionAgent with:
- Pieces Per Hour (PPH) tracking
- Task completion tracking
- Meshcat saving per iteration
- Proper action unnormalization using MPI LinearNormalizer
"""

import os
import numpy as np
import torch
import logging
import wandb
from typing import Optional, Dict

log = logging.getLogger(__name__)
from agent.finetune.train_ppo_diffusion_img_agent import TrainPPOImgDiffusionAgent


def _load_mpi_model_for_debug(checkpoint_path: str, normalizer_path: str, device: str = "cuda"):
    """
    Load the MPI model exactly as MPI does it, for debugging comparison.
    
    This loads:
    1. The workspace with full config
    2. The NormalizedViTDiffusionPolicy (wrapper that handles normalization)
    3. Sets the normalizer
    
    Returns the NormalizedViTDiffusionPolicy instance ready for inference.
    """
    import hydra
    from omegaconf import OmegaConf
    
    # Load checkpoint
    log.info(f"[DEBUG MPI] Loading MPI checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get the config from checkpoint
    cfg = checkpoint.get("cfg", None)
    if cfg is None:
        log.error("[DEBUG MPI] No cfg found in checkpoint!")
        return None
    
    # Log config structure
    log.info(f"[DEBUG MPI] Checkpoint config keys: {list(cfg.keys()) if hasattr(cfg, 'keys') else 'OmegaConf'}")
    
    # Get the policy config
    if hasattr(cfg, 'policy'):
        policy_cfg = cfg.policy
        log.info(f"[DEBUG MPI] Policy config: {OmegaConf.to_yaml(policy_cfg)[:500]}...")
    else:
        log.error("[DEBUG MPI] No policy config found!")
        return None
    
    # Instantiate the policy using hydra
    try:
        from diffusion_policy.policy.vit_diffusion_policy import NormalizedViTDiffusionPolicy
        
        # Create the policy from config
        policy = hydra.utils.instantiate(policy_cfg)
        log.info(f"[DEBUG MPI] Created policy: {type(policy)}")
        
        # Load state dict
        state_dict = checkpoint.get("state_dicts", {}).get("model", None)
        if state_dict is None:
            log.error("[DEBUG MPI] No model state_dict found in checkpoint!")
            return None
        
        # Strip "module." prefix from DDP training
        # MPI checkpoints are saved from DDP wrapped models, so keys have "module." prefix
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_key = key[7:]  # Remove "module." prefix
            else:
                new_key = key
            cleaned_state_dict[new_key] = value
        
        log.info(f"[DEBUG MPI] Converted {len(state_dict)} keys, stripped 'module.' prefix")
        log.info(f"[DEBUG MPI] Sample keys after conversion: {list(cleaned_state_dict.keys())[:5]}")
        
        # Load weights
        policy.load_state_dict(cleaned_state_dict)
        log.info(f"[DEBUG MPI] Loaded state dict successfully!")
        
        # Load normalizer
        normalizer = torch.load(normalizer_path, weights_only=False)
        policy.set_normalizer(normalizer)
        log.info(f"[DEBUG MPI] Set normalizer with keys: {list(normalizer.params_dict.keys())}")
        
        # Move to device and eval mode
        policy = policy.to(device)
        policy.eval()
        
        return policy
        
    except Exception as e:
        log.error(f"[DEBUG MPI] Failed to load MPI model: {e}")
        import traceback
        traceback.print_exc()
        return None


class TrainPPODiffusionTruck2DAgent(TrainPPOImgDiffusionAgent):
    """
    DPPO agent specialized for truck_2d task with custom metrics.
    
    CRITICAL: This agent handles action unnormalization using the MPI LinearNormalizer,
    which maps from the learned training data distribution to actual joint positions.
    The environment must have normalize_actions=False since we unnormalize here.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Truck_2d specific config
        self.save_meshcat = getattr(cfg.env, 'save_meshcat', False)
        self.approx_n_meshcats_saved = getattr(cfg.env, 'approx_n_meshcats_saved', 0)
        self.meshcat_dir = os.path.join(self.logdir, 'meshcats')
        if self.save_meshcat:
            os.makedirs(self.meshcat_dir, exist_ok=True)
            log.info(f"Meshcat saving enabled: will save ~{self.approx_n_meshcats_saved} per iteration to {self.meshcat_dir}")
            
        # Episode tracking for custom metrics
        self._episode_infos = []
        # Counter for meshcat saving
        self._meshcat_save_count = 0
        self._episodes_this_iteration = 0
        
        # Critic observation preprocessing settings
        # Allows critic to use fewer observation steps than policy (e.g., single-frame)
        self.critic_n_obs_steps = cfg.get('critic_n_obs_steps', self.n_cond_step)
        self.critic_img_cond_steps = cfg.get('critic_img_cond_steps', self.n_cond_step)
        log.info(f"Critic uses {self.critic_n_obs_steps} obs steps, {self.critic_img_cond_steps} img cond steps")
        
        # Validate that policy normalizer is loaded for action unnormalization
        if self.policy_normalizer is None:
            log.warning(
                "No policy_normalizer loaded! Actions will NOT be unnormalized correctly. "
                "Set policy_normalizer_path in config to load the MPI normalizer."
            )
        else:
            # Check that the normalizer has an 'action' key
            if hasattr(self.policy_normalizer, 'params_dict'):
                if 'action' not in self.policy_normalizer.params_dict:
                    log.warning(
                        f"Policy normalizer does not have 'action' key! "
                        f"Available keys: {list(self.policy_normalizer.params_dict.keys())}"
                    )
                else:
                    # Log the action normalizer stats for debugging
                    action_params = self.policy_normalizer.params_dict['action']
                    if 'input_stats' in action_params:
                        stats = action_params['input_stats']
                        log.info(
                            f"Action normalizer stats: "
                            f"min={stats['min'].cpu().numpy()}, "
                            f"max={stats['max'].cpu().numpy()}"
                        )
        
        # DEBUG: Load MPI model for comparison
        self._mpi_model = None  # Will be loaded on first use
        self._use_mpi_model_for_debug = cfg.get('use_mpi_model_for_debug', False)
        self._compare_with_mpi_model = cfg.get('compare_with_mpi_model', True)  # Always compare if checkpoint available
        self._mpi_checkpoint_path = getattr(cfg, 'policy_checkpoint_path', None)
        self._mpi_normalizer_path = getattr(cfg, 'policy_normalizer_path', None)
        
        if self._use_mpi_model_for_debug:
            log.info("[DEBUG] MPI model debugging enabled - will load and use MPI model directly")
            self._load_mpi_model()
        elif self._compare_with_mpi_model and self._mpi_checkpoint_path:
            log.info("[DEBUG] Will load MPI model for comparison (not for inference)")
            self._load_mpi_model()
    
    def _load_mpi_model(self):
        """Load the MPI model for debugging comparison."""
        if self._mpi_checkpoint_path and self._mpi_normalizer_path:
            self._mpi_model = _load_mpi_model_for_debug(
                self._mpi_checkpoint_path,
                self._mpi_normalizer_path,
                str(self.device)
            )
            if self._mpi_model is not None:
                log.info("[DEBUG] MPI model loaded successfully!")
            else:
                log.error("[DEBUG] Failed to load MPI model!")
        else:
            log.warning("[DEBUG] Cannot load MPI model - checkpoint_path or normalizer_path not set")
    
    def _predict_with_mpi_model(self, obs: dict) -> np.ndarray:
        """
        Run inference using the MPI model (NormalizedViTDiffusionPolicy).
        
        This method converts DPPO observation format to MPI format and
        runs inference through the MPI model for comparison.
        
        Args:
            obs: Dict with 'rgb' [B, T, C, H, W] and 'state' [B, T, D] as torch tensors
                 (already normalized)
        
        Returns:
            Unnormalized actions from MPI model [B, horizon, action_dim]
        """
        if self._mpi_model is None:
            raise RuntimeError("MPI model not loaded - call _load_mpi_model() first")
        
        # Convert DPPO format to MPI format
        # DPPO: {'rgb': [B,T,C,H,W], 'state': [B,T,D]} (state already normalized)
        # MPI: {'images': [B,T,C,H,W], 'robot_state': [B,T,D]} (state NOT normalized - model handles it)
        
        # IMPORTANT: MPI model's predict_action expects RAW observations
        # and normalizes internally. So we need to pass raw observations.
        # But DPPO already normalized the state. We need to UNNORMALIZE first.
        
        # Unnormalize state back to raw values for MPI model
        state = obs['state']  # Already normalized
        original_shape = state.shape
        state_flat = state.reshape(-1, state.shape[-1])
        
        # Unnormalize using the normalizer
        if self.policy_normalizer is not None:
            raw_state = self.policy_normalizer.unnormalize({'robot_state': state_flat})['robot_state']
        else:
            raw_state = state_flat
        raw_state = raw_state.reshape(original_shape)
        
        # Create MPI format observation dict (with raw state)
        mpi_obs = {
            'images': obs['rgb'],  # Images stay the same
            'robot_state': raw_state,  # Raw state for MPI model
        }
        
        # Run MPI model inference
        with torch.no_grad():
            results = self._mpi_model.predict_action(mpi_obs, use_ddim=False)
            # MPI model returns UNNORMALIZED actions
            action_pred = results['action_pred'].cpu().numpy()
        
        return action_pred
    
    def _save_debug_images(self, rgb: torch.Tensor, itr: int, step: int):
        """
        Save depth images to disk for visual inspection.
        
        Args:
            rgb: Tensor of shape [B, T, C, H, W] containing grayscale depth images
            itr: Current iteration number
            step: Current step number
        """
        import os
        from PIL import Image
        import numpy as np
        
        # Create debug images directory
        debug_dir = os.path.join(self.logdir, 'debug_images')
        os.makedirs(debug_dir, exist_ok=True)
        
        # Convert to numpy
        rgb_np = rgb.cpu().numpy()  # [B, T, C, H, W]
        
        # Save images for each environment and timestep
        n_envs = min(rgb_np.shape[0], 3)  # Save at most 3 environments
        n_timesteps = rgb_np.shape[1]
        
        for env_idx in range(n_envs):
            for t_idx in range(n_timesteps):
                # Extract single image [C, H, W] -> [H, W] for grayscale
                img = rgb_np[env_idx, t_idx, 0]  # First channel (grayscale)
                
                # Convert to 8-bit for saving (0-255)
                img_uint8 = (img * 255).astype(np.uint8)
                
                # Save as PNG
                filename = f"itr{itr}_step{step}_env{env_idx}_t{t_idx}.png"
                filepath = os.path.join(debug_dir, filename)
                Image.fromarray(img_uint8, mode='L').save(filepath)
        
        log.info(f"[DEBUG] Saved {n_envs * n_timesteps} debug images to {debug_dir}")
    
    def _normalize_observations(self, obs: dict) -> dict:
        """
        Normalize observations for the MPI/DPPO policy model.
        
        The DPPO VisionUnet1D expects:
        - state: (B, To, obs_dim) - will be flattened internally to (B, To*obs_dim)
        - rgb: (B, T_rgb, C, H, W) - passed as-is, internal processing handles it
        
        This function:
        - Normalizes state values to [-1, 1] using the MPI normalizer
        - Leaves rgb unchanged (already in [0, 1] range)
        
        Args:
            obs: Dict with 'rgb' [B, T, C, H, W] and 'state' [B, T, D] as torch tensors
        
        Returns:
            Dict with normalized observations (same shapes as input)
        """
        if self.policy_normalizer is None:
            log.warning("No policy normalizer - returning observations as-is!")
            return obs
        
        normalized_obs = {}
        
        # RGB: pass through unchanged (already in [0, 1] range)
        if 'rgb' in obs:
            normalized_obs['rgb'] = obs['rgb']
        
        # Normalize state using robot_state key from MPI normalizer
        if 'state' in obs:
            state = obs['state']
            original_shape = state.shape  # (B, T, D)
            
            # Flatten: (B, T, D) -> (B*T, D) for normalization
            state_flat = state.reshape(-1, state.shape[-1])
            
            # The MPI normalizer uses 'robot_state' key, not 'state'
            try:
                normalized = self.policy_normalizer.normalize({'robot_state': state_flat})
                normalized_state = normalized['robot_state']
            except KeyError:
                # Fallback: try 'state' key
                try:
                    normalized = self.policy_normalizer.normalize({'state': state_flat})
                    normalized_state = normalized['state']
                except KeyError:
                    log.warning("Policy normalizer has no 'robot_state' or 'state' key - using raw state")
                    normalized_state = state_flat
            
            # Reshape back to original shape: (B*T, D) -> (B, T, D)
            normalized_obs['state'] = normalized_state.reshape(original_shape)
        
        return normalized_obs
    
    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        Unnormalize actions using the MPI policy normalizer.
        
        The diffusion model outputs actions in normalized space (approximately [-1, 1]).
        This method converts them to actual joint positions using the learned
        normalizer from MPI pre-training.
        
        Args:
            actions: Normalized actions from diffusion model, shape (n_envs, act_steps, action_dim)
                    or (n_envs, horizon_steps, action_dim)
        
        Returns:
            Unnormalized actions (actual joint positions), same shape as input
        """
        if self.policy_normalizer is None:
            log.warning("No policy normalizer - returning actions as-is (this will likely fail!)")
            return actions
        
        # Convert to torch tensor
        actions_tensor = torch.from_numpy(actions).float().to(self.device)
        original_shape = actions_tensor.shape
        
        # Flatten for normalization: (n_envs, act_steps, action_dim) -> (n_envs * act_steps, action_dim)
        actions_flat = actions_tensor.reshape(-1, actions_tensor.shape[-1])
        
        # Unnormalize using the policy normalizer's action key
        try:
            # Use the LinearNormalizer's unnormalize method with dict input
            unnormalized = self.policy_normalizer.unnormalize({'action': actions_flat})
            unnormalized_actions = unnormalized['action']
        except Exception as e:
            log.error(f"Failed to unnormalize actions: {e}")
            log.error(f"Actions shape: {actions_flat.shape}")
            log.error(f"Normalizer keys: {list(self.policy_normalizer.params_dict.keys()) if hasattr(self.policy_normalizer, 'params_dict') else 'unknown'}")
            raise
        
        # Reshape back to original shape
        unnormalized_actions = unnormalized_actions.reshape(original_shape)
        
        return unnormalized_actions.cpu().numpy()
    
    def _preprocess_obs_for_critic(self, obs):
        """
        Preprocess observations for critic, extracting only the latest frame(s).
        
        This allows the critic to use single-frame observations while the policy
        uses multi-step observations, matching the MPI value network architecture.
        
        Args:
            obs: Dict with 'rgb' [B, T, C, H, W] and 'state' [B, T, D]
            
        Returns:
            Dict with 'rgb' [B, critic_img_cond_steps, C, H, W] and 
                       'state' [B, critic_n_obs_steps, D]
        """
        import torch
        
        critic_obs = {}
        
        # Extract latest frames for rgb
        if 'rgb' in obs:
            rgb = obs['rgb']
            if isinstance(rgb, torch.Tensor) and rgb.dim() == 5:
                # [B, T, C, H, W] - take latest critic_img_cond_steps frames
                critic_obs['rgb'] = rgb[:, -self.critic_img_cond_steps:]
            else:
                critic_obs['rgb'] = rgb
        
        # Extract latest state observations
        if 'state' in obs:
            state = obs['state']
            if isinstance(state, torch.Tensor) and state.dim() == 3:
                # [B, T, D] - take latest critic_n_obs_steps
                critic_obs['state'] = state[:, -self.critic_n_obs_steps:]
            else:
                critic_obs['state'] = state
                
        return critic_obs
        
    def _compute_custom_metrics(self, info_venv, step: int = -1):
        """
        Extract custom metrics from episode infos.
        
        Calculates:
        - avg_pieces_per_hour: Average PPH across completed episodes
        - avg_task_completion: Average task completion ratio
        - success_rate_custom: Success rate based on environment status
        
        Args:
            info_venv: List of info dicts from each environment
            step: Current step within iteration (for debug logging)
        """
        # Collect all completed episode infos from this step
        for env_idx, info in enumerate(info_venv):
            # Convert status to string (might be numpy array from vectorized env)
            status = info.get('status', 'unknown')
            if isinstance(status, np.ndarray):
                status = str(status.item()) if status.size == 1 else str(status[0])
            
            # Extract numeric values safely
            n_boxes_removed = info.get('n_boxes_removed', 0)
            n_boxes_total = info.get('n_boxes_total', 0)
            duration = info.get('duration', 0)
            if isinstance(n_boxes_removed, np.ndarray):
                n_boxes_removed = int(n_boxes_removed.item()) if n_boxes_removed.size == 1 else int(n_boxes_removed[0])
            if isinstance(n_boxes_total, np.ndarray):
                n_boxes_total = int(n_boxes_total.item()) if n_boxes_total.size == 1 else int(n_boxes_total[0])
            if isinstance(duration, np.ndarray):
                duration = float(duration.item()) if duration.size == 1 else float(duration[0])
            
            # Debug log for every step to trace environment flow
            log.debug(
                f"[Env {env_idx}] Step {step}: status={status}, "
                f"n_boxes_removed={n_boxes_removed}/{n_boxes_total}, "
                f"duration={duration:.2f}s"
            )
            
            # Check if episode ended (info has termination data)
            if 'is_success' in info or status in ['success', 'timeout', 'penetration_fail', 'tracking_fail', 'no_simulation']:
                self._episodes_this_iteration += 1
                
                # Extract is_success safely
                is_success = info.get('is_success', False)
                if isinstance(is_success, np.ndarray):
                    is_success = bool(is_success.item()) if is_success.size == 1 else bool(is_success[0])
                
                ep_info = {
                    'env_idx': env_idx,
                    'n_boxes_total': n_boxes_total,
                    'n_boxes_removed': n_boxes_removed,
                    'duration': duration,
                    'is_success': is_success,
                    'status': status,
                    'step': step,
                }
                self._episode_infos.append(ep_info)
                
                # Log episode result at INFO level
                log.info(
                    f"[Env {env_idx}] Episode ended at step {step}: "
                    f"status={status}, boxes_removed={n_boxes_removed}/{n_boxes_total}, "
                    f"duration={duration:.2f}s, success={is_success}"
                )
        
    def _aggregate_custom_metrics(self, timeout_duration: float = 100.0):
        """
        Aggregate custom metrics from all completed episodes in this iteration.
        """
        log.debug(f"Aggregating metrics from {len(self._episode_infos)} episodes")
        
        if not self._episode_infos:
            log.debug("No episodes completed this iteration")
            return {
                'avg_pieces_per_hour': 0.0,
                'avg_task_completion': 0.0,
                'custom_success_rate': 0.0,
                'n_episodes_completed': 0,
            }
        
        # Count status types for summary
        status_counts = {}
        pph_values = []
        tc_values = []
        n_success = 0
        
        for ep_info in self._episode_infos:
            n_total = ep_info.get('n_boxes_total', 1)
            n_removed = ep_info.get('n_boxes_removed', 0)
            is_success = ep_info.get('is_success', False)
            status = ep_info.get('status', 'unknown')
            
            # Count statuses
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Task completion
            if n_total > 0:
                tc_values.append(n_removed / n_total)
            
            # PPH calculation
            if is_success:
                duration = max(0.001, ep_info.get('duration', 0))
                n_success += 1
            else:
                duration = timeout_duration
            
            if duration > 0 and n_removed >= 0:
                pph = (n_removed / duration) * 3600  # pieces per hour
                pph_values.append(pph)
        
        metrics = {
            'avg_pieces_per_hour': float(np.mean(pph_values)) if pph_values else 0.0,
            'avg_task_completion': float(np.mean(tc_values)) if tc_values else 0.0,
            'custom_success_rate': n_success / len(self._episode_infos) if self._episode_infos else 0.0,
            'n_episodes_completed': len(self._episode_infos),
        }
        
        # Log iteration summary
        log.info(
            f"Iteration summary: {len(self._episode_infos)} episodes completed, "
            f"status breakdown: {status_counts}"
        )
        log.debug(
            f"Metrics: PPH={metrics['avg_pieces_per_hour']:.2f}, "
            f"task_completion={metrics['avg_task_completion']:.4f}, "
            f"success_rate={metrics['custom_success_rate']:.4f}"
        )
        
        # Clear episode infos for next iteration
        self._episode_infos = []
        
        return metrics

    def run(self):
        """Override run to add custom metric tracking."""
        from util.timer import Timer
        import math
        import einops
        import torch
        import pickle
        
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        
        while self.itr < self.n_train_itr:
            # Clear episode infos for this iteration
            self._episode_infos = []
            self._episodes_this_iteration = 0
            
            log.debug(f"=== Starting iteration {self.itr}/{self.n_train_itr} ===")

            # Prepare video and meshcat paths for each env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )
            
            # Select random environments to save meshcats for this iteration
            meshcat_env_indices = []
            if self.save_meshcat and self.approx_n_meshcats_saved > 0:
                n_to_save = min(self.approx_n_meshcats_saved, self.n_envs)
                meshcat_env_indices = list(np.random.choice(
                    self.n_envs, n_to_save, replace=False
                ))
                for env_ind in meshcat_env_indices:
                    meshcat_path = os.path.join(
                        self.meshcat_dir, f"itr-{self.itr}_env-{env_ind}.html"
                    )
                    options_venv[env_ind]["meshcat_path"] = meshcat_path
                    log.debug(f"Meshcat enabled for env {env_ind}: {meshcat_path}")

            # Define train or eval
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode
            
            log.debug(f"Iteration {self.itr}: eval_mode={eval_mode}, n_envs={self.n_envs}, n_steps={self.n_steps}")

            # Reset env before iteration
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                log.debug(f"Resetting all {self.n_envs} environments for iteration {self.itr}")
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                log.debug(f"Continuing from previous state, done_venv={done_venv}")
                firsts_trajs[0] = done_venv

            # Holders
            obs_trajs = {
                k: np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dims[k])
                )
                for k in self.obs_dims
            }
            chains_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            # Collect trajectories
            for step in range(self.n_steps):
                if step % 10 == 0:
                    log.info(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch.no_grad():
                    cond = {
                        key: torch.from_numpy(prev_obs_venv[key])
                        .float()
                        .to(self.device)
                        for key in self.obs_dims
                    }
                    log.info(f"before querying model: cond shape for all keys: {[cond[key].shape for key in cond.keys()]}")
                    # CRITICAL: Normalize observations before passing to model
                    # MPI policy expects normalized observations (robot_state in [-1, 1])
                    cond = self._normalize_observations(cond)
                    
                    # DEBUG: Option to use MPI model directly instead of DPPO model
                    if self._use_mpi_model_for_debug and self._mpi_model is not None:
                        # Use MPI model - it returns UNNORMALIZED actions directly
                        mpi_actions = self._predict_with_mpi_model(cond)
                        output_venv = mpi_actions
                        chains_venv = None
                        
                        # CRITICAL: Pass FULL prediction (all horizon_steps) for trajectory creation
                        # MPI uses the full prediction to create trajectories, not just act_steps.
                        # This ensures the trajectory has non-zero velocity at the end of executed steps,
                        # because the trajectory extends beyond act_steps (towards the remaining waypoints).
                        # Only act_steps worth of simulation is executed, but the trajectory is smoother.
                        action_venv = output_venv  # Use FULL prediction, not truncated!
                        action_venv_unnorm = action_venv  # Already unnormalized!
                        
                        if step == 0:
                            # Debug observation info
                            log.info(
                                f"[DEBUG MPI MODEL] Using MPI model for inference\n"
                                f"  Observation shapes: rgb={cond['rgb'].shape}, state={cond['state'].shape}\n"
                                f"  RGB range: [{cond['rgb'].min():.3f}, {cond['rgb'].max():.3f}]\n"
                                f"  State (normalized) [0,0,:]: {cond['state'][0, 0, :].cpu().numpy()}\n"
                                f"  Full prediction shape: {action_venv_unnorm.shape} (horizon_steps={action_venv_unnorm.shape[1]})\n"
                                f"  MPI action[0,0,:4]: {action_venv_unnorm[0, 0, :4]}\n"
                                f"  MPI action[0,{self.act_steps-1},:4] (last executed): {action_venv_unnorm[0, self.act_steps-1, :4]}\n"
                                f"  MPI action[0,-1,:4] (last predicted): {action_venv_unnorm[0, -1, :4]}"
                            )
                    else:
                        # Use DPPO model (default)
                        samples = self.model(
                            cond=cond,
                            deterministic=eval_mode,
                            return_chain=True,
                        )
                        output_venv = samples.trajectories.cpu().numpy()
                        chains_venv = samples.chains.cpu().numpy() if samples.chains is not None else None
                        # action_venv = output_venv[:, : self.act_steps]
                        action_venv = output_venv # use full prediction
                        # CRITICAL: Unnormalize actions using MPI policy normalizer
                        # The diffusion model outputs normalized actions, but the environment
                        # expects raw joint positions (with normalize_actions=False)
                        action_venv_unnorm = self._unnormalize_actions(action_venv)
                        

                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv_unnorm)
                )
                done_venv = terminated_venv | truncated_venv
                
                # Debug log for environment step results
                log.debug(
                    f"Step {step}: rewards={reward_venv}, terminated={terminated_venv}, "
                    f"truncated={truncated_venv}"
                )
                
                # Log meshcat saves (environment auto-saves meshcat when episode ends)
                for env_ind in meshcat_env_indices:
                    if done_venv[env_ind]:
                        status = info_venv[env_ind].get("status", "unknown")
                        log.info(f"Episode ended for env {env_ind} with status: {status} (meshcat auto-saved)")
                
                # Track custom metrics from info
                self._compute_custom_metrics(info_venv, step=step)
                
                for k in obs_trajs:
                    obs_trajs[k][step] = prev_obs_venv[k]
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv

                prev_obs_venv = obs_venv
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            # Summarize episode rewards
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # Compute custom truck_2d metrics
            custom_metrics = self._aggregate_custom_metrics(
                timeout_duration=self.max_episode_steps * 0.1  # control_timestep
            )

            # Update models (copied from parent class with minor modifications)
            if not eval_mode:
                with torch.no_grad():
                    obs_trajs["rgb"] = (
                        torch.from_numpy(obs_trajs["rgb"]).float().to(self.device)
                    )
                    obs_trajs["state"] = (
                        torch.from_numpy(obs_trajs["state"]).float().to(self.device)
                    )
                    if self.augment:
                        rgb = einops.rearrange(
                            obs_trajs["rgb"],
                            "s e t c h w -> (s e t) c h w",
                        )
                        rgb = self.aug(rgb)
                        obs_trajs["rgb"] = einops.rearrange(
                            rgb,
                            "(s e t) c h w -> s e t c h w",
                            s=self.n_steps,
                            e=self.n_envs,
                        )

                    # Calculate value and logprobs
                    num_split = math.ceil(
                        self.n_envs * self.n_steps / self.logprob_batch_size
                    )
                    obs_ts = [{} for _ in range(num_split)]
                    for k in obs_trajs:
                        obs_k = einops.rearrange(
                            obs_trajs[k],
                            "s e ... -> (s e) ...",
                        )
                        obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                        for i, obs_t in enumerate(obs_ts_k):
                            obs_ts[i][k] = obs_t
                    values_trajs = np.empty((0, self.n_envs))
                    for obs in obs_ts:
                        # Preprocess obs for critic (may use fewer timesteps)
                        critic_obs = self._preprocess_obs_for_critic(obs)
                        values = (
                            self.model.critic(critic_obs, no_augment=True)
                            .cpu()
                            .numpy()
                            .flatten()
                        )
                        values_trajs = np.vstack(
                            (values_trajs, values.reshape(-1, self.n_envs))
                        )
                    chains_t = einops.rearrange(
                        torch.from_numpy(chains_trajs).float().to(self.device),
                        "s e t h d -> (s e) t h d",
                    )
                    chains_ts = torch.split(chains_t, self.logprob_batch_size, dim=0)
                    logprobs_trajs = np.empty(
                        (
                            0,
                            self.model.ft_denoising_steps,
                            self.horizon_steps,
                            self.action_dim,
                        )
                    )
                    for obs, chains in zip(obs_ts, chains_ts):
                        logprobs = self.model.get_logprobs(obs, chains).cpu().numpy()
                        logprobs_trajs = np.vstack(
                            (
                                logprobs_trajs,
                                logprobs.reshape(-1, *logprobs_trajs.shape[1:]),
                            )
                        )

                    # Normalize reward
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_transpose.T

                    # GAE
                    obs_venv_ts = {
                        key: torch.from_numpy(obs_venv[key]).float().to(self.device)
                        for key in self.obs_dims
                    }
                    # Preprocess for critic (may use fewer timesteps)
                    critic_obs_venv = self._preprocess_obs_for_critic(obs_venv_ts)
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextvalues = (
                                self.model.critic(critic_obs_venv, no_augment=True)
                                .reshape(1, -1)
                                .cpu()
                                .numpy()
                            )
                        else:
                            nextvalues = values_trajs[t + 1]
                        nonterminal = 1.0 - terminated_trajs[t]
                        delta = (
                            reward_trajs[t] * self.reward_scale_const
                            + self.gamma * nextvalues * nonterminal
                            - values_trajs[t]
                        )
                        advantages_trajs[t] = lastgaelam = (
                            delta
                            + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                        )
                    returns_trajs = advantages_trajs + values_trajs

                # Prepare for updates
                obs_k = {
                    k: einops.rearrange(
                        obs_trajs[k],
                        "s e ... -> (s e) ...",
                    )
                    for k in obs_trajs
                }
                chains_k = einops.rearrange(
                    torch.tensor(chains_trajs, device=self.device).float(),
                    "s e t h d -> (s e) t h d",
                )
                returns_k = (
                    torch.tensor(returns_trajs, device=self.device).float().reshape(-1)
                )
                values_k = (
                    torch.tensor(values_trajs, device=self.device).float().reshape(-1)
                )
                advantages_k = (
                    torch.tensor(advantages_trajs, device=self.device).float().reshape(-1)
                )
                logprobs_k = torch.tensor(logprobs_trajs, device=self.device).float()

                # Policy and critic updates
                total_steps = self.n_steps * self.n_envs * self.model.ft_denoising_steps
                clipfracs = []
                for update_epoch in range(self.update_epochs):
                    flag_break = False
                    inds_k = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]
                        batch_inds_b, denoising_inds_b = torch.unravel_index(
                            inds_b,
                            (self.n_steps * self.n_envs, self.model.ft_denoising_steps),
                        )
                        obs_b = {k: obs_k[k][batch_inds_b] for k in obs_k}
                        # Preprocess obs for critic (may use fewer timesteps)
                        critic_obs_b = self._preprocess_obs_for_critic(obs_b)
                        chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]
                        chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]
                        returns_b = returns_k[batch_inds_b]
                        values_b = values_k[batch_inds_b]
                        advantages_b = advantages_k[batch_inds_b]
                        logprobs_b = logprobs_k[batch_inds_b, denoising_inds_b]

                        # Get loss (pass critic_obs for single-frame critic)
                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            eta,
                        ) = self.model.loss(
                            obs_b,
                            chains_prev_b,
                            chains_next_b,
                            denoising_inds_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                            reward_horizon=self.reward_horizon,
                            critic_obs=critic_obs_b,  # Single-frame for MPI critic
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )
                        clipfracs += [clipfrac]

                        # Update policy and critic
                        loss.backward()
                        if (batch + 1) % self.grad_accumulate == 0:
                            if self.itr >= self.n_critic_warmup_itr:
                                if self.max_grad_norm is not None:
                                    torch.nn.utils.clip_grad_norm_(
                                        self.model.actor_ft.parameters(),
                                        self.max_grad_norm,
                                    )
                                self.actor_optimizer.step()
                                if (
                                    self.learn_eta
                                    and batch % self.eta_update_interval == 0
                                ):
                                    self.eta_optimizer.step()
                            self.critic_optimizer.step()
                            self.actor_optimizer.zero_grad()
                            self.critic_optimizer.zero_grad()
                            if self.learn_eta:
                                self.eta_optimizer.zero_grad()
                            log.info(f"run grad update at batch {batch}")
                            log.info(
                                f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}"
                            )

                            if (
                                self.target_kl is not None
                                and approx_kl > self.target_kl
                                and self.itr >= self.n_critic_warmup_itr
                            ):
                                flag_break = True
                                break
                    if flag_break:
                        break

                # Explained variance
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

            # Update lr
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log metrics (with custom truck_2d metrics)
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                    **custom_metrics,
                }
            )
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    log.info(
                        f"      PPH {custom_metrics['avg_pieces_per_hour']:8.2f} | task_completion {custom_metrics['avg_task_completion']:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                                # Custom metrics
                                "avg_pieces_per_hour - eval": custom_metrics['avg_pieces_per_hour'],
                                "avg_task_completion - eval": custom_metrics['avg_task_completion'],
                                "custom_success_rate - eval": custom_metrics['custom_success_rate'],
                            },
                            step=self.itr,
                            commit=False,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | eta {eta:8.4f} | t:{time:8.4f}"
                    )
                    log.info(
                        f"      PPH {custom_metrics['avg_pieces_per_hour']:8.2f} | task_completion {custom_metrics['avg_task_completion']:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "loss": loss,
                                "pg loss": pg_loss,
                                "value loss": v_loss,
                                "bc loss": bc_loss,
                                "eta": eta,
                                "approx kl": approx_kl,
                                "ratio": ratio,
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                                "diffusion - min sampling std": diffusion_min_sampling_std,
                                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                                "critic lr": self.critic_optimizer.param_groups[0]["lr"],
                                # Custom metrics
                                "avg_pieces_per_hour - train": custom_metrics['avg_pieces_per_hour'],
                                "avg_task_completion - train": custom_metrics['avg_task_completion'],
                                "custom_success_rate - train": custom_metrics['custom_success_rate'],
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
