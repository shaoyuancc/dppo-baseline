"""
DPPO fine-tuning for truck_2d with custom metrics.

Extends TrainPPOImgDiffusionAgent with:
- Pieces Per Hour (PPH) tracking
- Task completion tracking
- Meshcat saving per iteration
- Proper action unnormalization using MPI LinearNormalizer

For MPI model integration, use config with:
    model:
      _target_: model.diffusion.mpi_ppo_diffusion.MPIPPODiffusion.from_mpi_checkpoints
      ...
"""

import os
import numpy as np
import torch
import logging
import wandb

log = logging.getLogger(__name__)
from agent.finetune.train_ppo_diffusion_img_agent import TrainPPOImgDiffusionAgent


class TrainPPODiffusionTruck2DAgent(TrainPPOImgDiffusionAgent):
    """
    DPPO agent specialized for truck_2d task with custom metrics.
    
    The model is created via hydra config. For MPI models, use:
        model._target_: model.diffusion.mpi_ppo_diffusion.MPIPPODiffusion.from_mpi_checkpoints
    
    CRITICAL: This agent handles action unnormalization using the MPI LinearNormalizer.
    The environment must have normalize_actions=False since we unnormalize here.
    """

    def __init__(self, cfg):
        # Call parent __init__ (creates environment, model via hydra, etc.)
        super().__init__(cfg)
        
        # Truck_2d specific config
        self.save_meshcat = getattr(cfg.env, 'save_meshcat', False)
        self.meshcat_save_freq = getattr(cfg.env, 'meshcat_save_freq', 1)  # Default: every iteration
        self.approx_n_meshcats_saved = getattr(cfg.env, 'approx_n_meshcats_saved', 0)
        self.meshcat_dir = os.path.join(self.logdir, 'meshcats')
        if self.save_meshcat:
            os.makedirs(self.meshcat_dir, exist_ok=True)
            log.info(f"Meshcat saving enabled: will save ~{self.approx_n_meshcats_saved} every {self.meshcat_save_freq} iterations to {self.meshcat_dir}")
            
        # Get control_timestep from config for timeout duration calculation in metrics
        # This is used to compute the actual episode timeout duration in seconds
        env_specific = getattr(cfg.env, 'specific', {})
        self.control_timestep = env_specific.get('control_timestep', 0.1)
        log.info(f"Control timestep for metrics: {self.control_timestep}s")
            
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
        
        # Environment reinitialization frequency (to reclaim leaked Drake memory)
        # Set to 0 to disable, 1 to reinit every iteration, N to reinit every N iterations
        self.env_reinit_freq = cfg.env.get('reinit_freq', 0)
        if self.env_reinit_freq > 0:
            log.info(f"Environment reinitialization enabled: every {self.env_reinit_freq} iteration(s)")
        
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
    
    def _compute_custom_metrics(self, info_venv, done_venv, step: int = -1):
        """
        Extract custom metrics from episode infos.
        
        Calculates:
        - avg_pieces_per_hour: Average PPH across completed episodes
        - avg_task_completion: Average task completion ratio
        - custom_success_rate: Success rate based on whether all boxes were removed
        
        Args:
            info_venv: List of info dicts from each environment
            done_venv: Boolean array indicating which environments finished an episode
            step: Current step within iteration (for debug logging)
        """
        def get_latest(value, default):
            """Extract the most recent value from potentially stacked info arrays."""
            if isinstance(value, np.ndarray):
                return value.flat[-1]  # Last element, works for any shape
            return value if value is not None else default
        
        # Collect all completed episode infos from this step
        for env_idx, info in enumerate(info_venv):
            episode_ended = bool(done_venv[env_idx])
            
            # Extract the most recent values from info
            # (MultiStep wrapper stacks last n_obs_steps values into arrays)
            status = str(get_latest(info.get('status'), 'unknown'))
            n_boxes_removed = int(get_latest(info.get('n_boxes_removed'), 0))
            n_boxes_total = int(get_latest(info.get('n_boxes_total'), 0))
            duration = float(get_latest(info.get('duration'), 0.0))
            is_success = bool(get_latest(info.get('is_success'), False))
            
            # Debug log for every step to trace environment flow
            log.debug(
                f"[Env {env_idx}] Step {step}: status={status}, "
                f"n_boxes_removed={n_boxes_removed}/{n_boxes_total}, "
                f"duration={duration:.2f}s"
            )
            
            # Only record episode info when the episode actually ended
            if episode_ended:
                self._episodes_this_iteration += 1
                
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
            # Reinitialize environments periodically to reclaim leaked Drake memory
            # This must happen BEFORE the memory log to measure the effect
            if self.env_reinit_freq > 0 and self.itr > 0 and self.itr % self.env_reinit_freq == 0:
                self._reinitialize_venv()
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
            # Only save meshcats at the specified frequency (like render.freq)
            meshcat_env_indices = []
            if self.save_meshcat and self.approx_n_meshcats_saved > 0 and self.itr % self.meshcat_save_freq == 0:
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
            # IMPORTANT: Use float32 to match observation dtype and avoid temporary copies
            obs_trajs = {
                k: np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dims[k]),
                    dtype=np.float32
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
                ),
                dtype=np.float32
            )
            terminated_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
            reward_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

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
                    # NOTE: Do NOT normalize here - MPIPolicyActorWrapper._encode_obs handles
                    # normalization internally, matching MPI's predict_action behavior.
                    # The training path (get_logprobs) also passes raw observations.
                    
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        return_chain=True,
                    )
                    output_venv = samples.trajectories.cpu().numpy()
                    chains_venv = samples.chains.cpu().numpy() if samples.chains is not None else None
                    action_venv = output_venv[:, : self.act_steps]
                    
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
                    f"Agent Step {step}: rewards={reward_venv}, terminated={terminated_venv}, "
                    f"truncated={truncated_venv}"
                )
                
                # Log meshcat saves (environment auto-saves meshcat when episode ends)
                for env_ind in meshcat_env_indices:
                    if done_venv[env_ind]:
                        status = info_venv[env_ind].get("status", "unknown")
                        log.info(f"Episode ended for env {env_ind} with status: {status} (meshcat auto-saved)")
                
                # Track custom metrics from info (only for episodes that ended)
                self._compute_custom_metrics(info_venv, done_venv, step=step)
                
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
            # timeout_duration = max_episode_steps * control_timestep (in seconds)
            custom_metrics = self._aggregate_custom_metrics(
                timeout_duration=self.max_episode_steps * self.control_timestep
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
                
                # Diagnostic logging for value function debugging
                returns_mean = float(np.mean(y_true))
                returns_std = float(np.std(y_true))
                values_mean = float(np.mean(y_pred))
                values_std = float(np.std(y_pred))
                advantages_mean = float(advantages_k.mean().cpu().numpy())
                advantages_std = float(advantages_k.std().cpu().numpy())
                log.info(
                    f"[Critic Diagnostics] Returns: mean={returns_mean:.4f}, std={returns_std:.4f} | "
                    f"Values: mean={values_mean:.4f}, std={values_std:.4f} | "
                    f"Advantages: mean={advantages_mean:.4f}, std={advantages_std:.4f}"
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
                                # Custom truck_2d metrics
                                "avg_pieces_per_hour - eval": custom_metrics['avg_pieces_per_hour'],
                                "avg_task_completion - eval": custom_metrics['avg_task_completion'],
                                "custom_success_rate - eval": custom_metrics['custom_success_rate'],
                                "n_episodes_completed - eval": custom_metrics['n_episodes_completed'],
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
                                # Custom truck_2d metrics
                                "avg_pieces_per_hour - train": custom_metrics['avg_pieces_per_hour'],
                                "avg_task_completion - train": custom_metrics['avg_task_completion'],
                                "custom_success_rate - train": custom_metrics['custom_success_rate'],
                                "n_episodes_completed - train": custom_metrics['n_episodes_completed'],
                                # Critic diagnostic metrics (for debugging value collapse)
                                "critic/returns_mean": returns_mean,
                                "critic/returns_std": returns_std,
                                "critic/values_mean": values_mean,
                                "critic/values_std": values_std,
                                "critic/advantages_mean": advantages_mean,
                                "critic/advantages_std": advantages_std,
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1

