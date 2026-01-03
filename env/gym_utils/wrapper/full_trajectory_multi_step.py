"""
Full trajectory multi-step wrapper for environments with smooth trajectory execution.

This wrapper is designed for environments that execute a full trajectory at once
(like MPI's PiecewisePolynomial trajectories) rather than stepping through actions
one at a time.

Key differences from MultiStep:
- Passes the entire action sequence to the environment as a 2D array
- Calls env.step_full_trajectory() instead of iterating through env.step()
- Handles list of observations returned from full trajectory execution
- Properly stacks observation history from multiple sub-steps

Usage:
    Use this wrapper instead of MultiStep when you want smooth trajectory execution.
    Configure in yaml: wrappers: { full_trajectory_multi_step: {...} }
"""

import gym
from typing import Optional
from gym import spaces
import numpy as np
from collections import defaultdict, deque


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x, axis=0), n, axis=0)


def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype,
    )


def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f"Unsupported space type {type(space)}")


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])


def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def stack_last_n_obs(all_obs, n_steps):
    """Apply padding to stack last n observations."""
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad with first available observation
        result[:start_idx] = result[start_idx]
    return result


class FullTrajectoryMultiStep(gym.Wrapper):
    """
    Wrapper for environments that execute full trajectories at once.
    
    Instead of iterating through actions one at a time (like MultiStep), this
    wrapper passes the entire action sequence to the environment which creates
    a smooth PiecewisePolynomial trajectory over all waypoints.
    
    IMPORTANT: This wrapper expects the FULL horizon trajectory from the policy,
    not just act_steps. The environment uses all horizon points to create a smooth
    trajectory but only executes act_steps worth of simulation. This matches MPI's
    behavior where the full horizon is used for smooth velocity at boundaries.
    
    The environment must implement step_full_trajectory(actions) method that:
    - Takes a 2D action array of shape (n_horizon_steps, action_dim)
    - Returns (observations_list, total_reward, done, info)
      where observations_list contains one observation per executed step
    
    Args:
        env: The wrapped environment (must have step_full_trajectory method)
        n_obs_steps: Number of observation history steps to stack
        n_action_steps: Number of action steps to execute (for tracking)
        n_horizon_steps: Number of horizon steps (full policy output size)
                        If not set, defaults to n_action_steps for backwards compatibility
        max_episode_steps: Maximum steps per episode (for truncation)
        reward_agg_method: How to aggregate rewards (default: "sum", but env already sums)
        prev_action: Whether to track previous actions
        reset_within_step: Whether to auto-reset when episode ends
        verbose: Enable debug logging
    """

    def __init__(
        self,
        env,
        n_obs_steps=1,
        n_action_steps=1,
        n_horizon_steps=None,  # Full horizon from policy (for smooth trajectory creation)
        max_episode_steps=None,
        reward_agg_method="sum",  # Not used - env already aggregates
        prev_action=True,
        reset_within_step=False,
        pass_full_observations=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__(env)
        self._single_action_space = env.action_space
        
        # Use n_horizon_steps for action space if provided, otherwise fall back to n_action_steps
        # This allows passing the full policy horizon to create smooth trajectories
        self.n_horizon_steps = n_horizon_steps if n_horizon_steps is not None else n_action_steps
        self._action_space = repeated_space(env.action_space, self.n_horizon_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps  # Still track for observation/reward handling
        self.reward_agg_method = reward_agg_method
        self.prev_action = prev_action
        self.reset_within_step = reset_within_step
        self.pass_full_observations = pass_full_observations
        self.verbose = verbose

        # Verify the environment supports full trajectory mode
        if not hasattr(env, 'step_full_trajectory'):
            raise ValueError(
                f"Environment {type(env).__name__} does not have step_full_trajectory method. "
                "FullTrajectoryMultiStep requires an environment that supports full trajectory execution."
            )

        # Initialize attributes that are properly set in reset()
        self.obs = None
        self.action = None
        self.info = None
        self.cnt = 0

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        """Resets the environment."""
        if options is None:
            options = {}
        obs = self.env.reset(
            seed=seed,
            options=options,
            return_info=return_info,
        )
        # Keep enough history for both obs stacking and action tracking
        maxlen = max(self.n_obs_steps + 1, self.n_action_steps)
        self.obs = deque([obs], maxlen=maxlen)
        if self.prev_action:
            self.action = deque(
                [self._single_action_space.sample()], maxlen=self.n_obs_steps
            )
        self.info = defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))

        self.cnt = 0
        return self._get_obs(self.n_obs_steps)

    def step(self, action):
        """
        Execute full trajectory of actions.
        
        The environment receives the FULL horizon trajectory (n_horizon_steps) which
        it uses to create a smooth PiecewisePolynomial. Only n_action_steps worth of
        simulation is executed, but having the full horizon ensures smooth velocity
        at the trajectory boundaries (matching MPI's behavior).
        
        Args:
            action: (n_horizon_steps, action_dim) array - full policy horizon output
            
        Returns:
            observation: Stacked observations (n_obs_steps, ...) 
            reward: Total reward from trajectory execution
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Info dict with trajectory execution details
        """
        if action.ndim == 1:
            # Single action - expand to 2D
            action = action[None]
        
        # Call environment's full trajectory execution
        observations_list, total_reward, done, info = self.env.step_full_trajectory(action)
        
        # Ensure observations_list is not empty (defensive check)
        if not observations_list:
            # If no observations returned, use the last known observation
            if len(self.obs) > 0:
                observations_list = [self.obs[-1]]
            else:
                raise RuntimeError("step_full_trajectory returned empty observations and no previous observation available")
        
        # Get number of steps actually executed
        n_executed = len(observations_list)
        
        # Update step counter
        self.cnt += n_executed
        
        # Add all observations to history
        for obs in observations_list:
            self.obs.append(obs)
        
        # Track actions (use last action for each step)
        if self.prev_action and self.action is not None:
            for i in range(min(n_executed, len(action))):
                self.action.append(action[i])
        
        # Add info entries (for compatibility with MultiStep info stacking)
        self._add_info(info)
        
        # Determine termination vs truncation
        if "TimeLimit.truncated" in info:
            truncated = info["TimeLimit.truncated"]
            terminated = done and not truncated
        else:
            terminated = done
            truncated = (
                self.max_episode_steps is not None
                and self.cnt >= self.max_episode_steps
                and not terminated
            )
        
        done = terminated or truncated
        
        # Get stacked observation for return
        observation = self._get_obs(self.n_obs_steps)
        
        # Prepare info - only include scalar values to avoid stacking issues
        # Don't use dict_take_last_n which can cause shape issues with arrays
        return_info = {}
        for key, value in self.info.items():
            if len(value) > 0:
                # Take the latest value only (not stacked)
                return_info[key] = value[-1]
        
        if self.pass_full_observations:
            return_info["full_obs"] = self._get_obs(n_executed)
        
        # Handle reset within step (like MultiStep)
        if self.reset_within_step and done:
            if truncated:
                return_info["final_obs"] = observation
            
            # Reset for next episode
            observation = self.reset()
            if self.verbose:
                print("Reset env within wrapper (full trajectory mode).")
        
        return observation, total_reward, terminated, truncated, return_info

    def _get_obs(self, n_steps=1):
        """
        Get stacked observations.
        
        Output shape: (n_steps,) + obs_shape
        """
        assert len(self.obs) > 0
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs([obs[key] for obs in self.obs], n_steps)
            return result
        else:
            raise RuntimeError("Unsupported space type")

    def get_prev_action(self, n_steps=None):
        if n_steps is None:
            n_steps = self.n_obs_steps - 1  # exclude current step
        assert len(self.action) > 0
        return stack_last_n_obs(self.action, n_steps)

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

