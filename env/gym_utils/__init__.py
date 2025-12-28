import os
import json

try:
    from collections.abc import Iterable
except ImportError:
    Iterable = (tuple, list)


# List of valid kwargs for truck_2d environment variants
_TRUCK_2D_ENV_KWARGS = [
    'problem_file', 'problem_index_range',
    'control_timestep', 'physics_timestep', 'render_mode',
    'max_suction_distance', 'max_suction_angle',
    'max_penetration_distance', 'enable_tracking_failure',
    'enable_penetration_failure', 'init_x',
    'reward_config',  # Reward configuration dict
    'max_boxes',  # Limit number of boxes per problem
    'normalize_actions',  # Whether env expects normalized actions
    'full_trajectory_mode',  # Use full trajectory execution like MPI
    'act_steps',  # Number of steps to execute in full trajectory mode
]


def _create_truck_2d_vectorized_env(
    env_class,
    num_envs,
    asynchronous,
    wrappers,
    action_dim,
    shape_meta,
    max_episode_steps,
    act_steps,
    **kwargs
):
    """
    Factory function for creating vectorized truck_2d environment variants.
    
    This handles TruckUnload2DEnv, TruckUnload2DReachEnv, TruckUnload2DGraspEnv,
    and any future variants that inherit from TruckUnload2DEnv.
    
    Args:
        env_class: The environment class to instantiate
        num_envs: Number of parallel environments
        asynchronous: Whether to use AsyncVectorEnv (True) or SyncVectorEnv (False)
        wrappers: Dictionary of wrapper classes and their arguments
        action_dim: Action dimension for dummy env
        shape_meta: Shape metadata for observation spaces
        max_episode_steps: Maximum steps per episode
        act_steps: Number of action steps to execute
        **kwargs: Additional environment-specific kwargs
    
    Returns:
        Vectorized environment (AsyncVectorEnv or SyncVectorEnv)
    """
    from gym import spaces
    from env.gym_utils.async_vector_env import AsyncVectorEnv
    from env.gym_utils.sync_vector_env import SyncVectorEnv
    from env.gym_utils.wrapper import wrapper_dict
    
    # Extract env-specific kwargs
    env_kwargs = {k: v for k, v in kwargs.items() if k in _TRUCK_2D_ENV_KWARGS}
    
    # Add max_episode_steps and act_steps (passed as direct parameters)
    if max_episode_steps is not None:
        env_kwargs['max_episode_steps'] = max_episode_steps
    if act_steps is not None:
        env_kwargs['act_steps'] = act_steps
    
    def _make_env():
        env = env_class(**env_kwargs)
        if wrappers is not None:
            for wrapper, args in wrappers.items():
                env = wrapper_dict[wrapper](env, **args)
        return env
    
    def dummy_env_fn():
        """Create dummy env for space introspection."""
        import gym
        import numpy as np
        from env.gym_utils.wrapper.multi_step import MultiStep
        
        env = gym.Env()
        observation_space = spaces.Dict()
        if shape_meta is not None:
            for key, value in shape_meta["obs"].items():
                shape = tuple(value["shape"])
                if key.endswith("rgb"):
                    min_value, max_value = 0, 1
                elif key.endswith("state"):
                    min_value, max_value = -np.inf, np.inf
                else:
                    min_value, max_value = -1, 1
                observation_space[key] = spaces.Box(
                    low=min_value,
                    high=max_value,
                    shape=shape,
                    dtype=np.float32,
                )
        env.observation_space = observation_space
        env.action_space = spaces.Box(-1, 1, shape=(action_dim,), dtype=np.float32)
        env.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 10,
        }
        n_obs_steps = wrappers.multi_step.n_obs_steps if hasattr(wrappers, 'multi_step') else 1
        return MultiStep(env=env, n_obs_steps=n_obs_steps)
    
    env_fns = [_make_env for _ in range(num_envs)]
    return (
        AsyncVectorEnv(
            env_fns,
            dummy_env_fn=dummy_env_fn,
            # Memory optimization: Don't deepcopy observations since with shared_memory=True
            # (the default), observations are already read from shared memory into a buffer.
            copy=False,
        )
        if asynchronous
        else SyncVectorEnv(env_fns)
    )


def make_async(
    id,
    num_envs=1,
    asynchronous=True,
    wrappers=None,
    render=False,
    obs_dim=23,
    action_dim=7,
    env_type=None,
    max_episode_steps=None,
    # below for furniture only
    gpu_id=0,
    headless=True,
    record=False,
    normalization_path=None,
    furniture="one_leg",
    randomness="low",
    obs_steps=1,
    act_steps=8,
    sparse_reward=False,
    # below for robomimic only
    robomimic_env_cfg_path=None,
    use_image_obs=False,
    render_offscreen=False,
    reward_shaping=False,
    shape_meta=None,
    **kwargs,
):
    """Create a vectorized environment from multiple copies of an environment,
    from its id.

    Parameters
    ----------
    id : str
        The environment ID. This must be a valid ID from the registry.

    num_envs : int
        Number of copies of the environment.

    asynchronous : bool
        If `True`, wraps the environments in an :class:`AsyncVectorEnv` (which uses
        `multiprocessing`_ to run the environments in parallel). If ``False``,
        wraps the environments in a :class:`SyncVectorEnv`.

    wrappers : dictionary, optional
        Each key is a wrapper class, and each value is a dictionary of arguments

    Returns
    -------
    :class:`gym.vector.VectorEnv`
        The vectorized environment.

    Example
    -------
    >>> env = gym.vector.make('CartPole-v1', num_envs=3)
    >>> env.reset()
    array([[-0.04456399,  0.04653909,  0.01326909, -0.02099827],
           [ 0.03073904,  0.00145001, -0.03088818, -0.03131252],
           [ 0.03468829,  0.01500225,  0.01230312,  0.01825218]],
          dtype=float32)
    """

    # Support for truck_2d environment variants (full unloading task)
    if env_type == "truck_2d":
        from dppo_baseline.env.truck_2d_env import TruckUnload2DEnv
        return _create_truck_2d_vectorized_env(
            env_class=TruckUnload2DEnv,
            num_envs=num_envs,
            asynchronous=asynchronous,
            wrappers=wrappers,
            action_dim=action_dim,
            shape_meta=shape_meta,
            max_episode_steps=max_episode_steps,
            act_steps=act_steps,
            **kwargs,
        )

    # Support for truck_2d_reach environment (simplified reach task)
    if env_type == "truck_2d_reach":
        from dppo_baseline.env.truck_2d_reach_env import TruckUnload2DReachEnv
        return _create_truck_2d_vectorized_env(
            env_class=TruckUnload2DReachEnv,
            num_envs=num_envs,
            asynchronous=asynchronous,
            wrappers=wrappers,
            action_dim=action_dim,
            shape_meta=shape_meta,
            max_episode_steps=max_episode_steps,
            act_steps=act_steps,
            **kwargs,
        )

    # Support for truck_2d_grasp environment (grasp task - suction success)
    if env_type == "truck_2d_grasp":
        from dppo_baseline.env.truck_2d_grasp_env import TruckUnload2DGraspEnv
        return _create_truck_2d_vectorized_env(
            env_class=TruckUnload2DGraspEnv,
            num_envs=num_envs,
            asynchronous=asynchronous,
            wrappers=wrappers,
            action_dim=action_dim,
            shape_meta=shape_meta,
            max_episode_steps=max_episode_steps,
            act_steps=act_steps,
            **kwargs,
        )

    if env_type == "furniture":
        from furniture_bench.envs.observation import DEFAULT_STATE_OBS
        from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
        from env.gym_utils.wrapper.furniture import FurnitureRLSimEnvMultiStepWrapper

        env = FurnitureRLSimEnv(
            act_rot_repr="rot_6d",
            action_type="pos",
            april_tags=False,
            concat_robot_state=True,
            ctrl_mode="diffik",
            obs_keys=DEFAULT_STATE_OBS,
            furniture=furniture,
            gpu_id=gpu_id,
            headless=headless,
            num_envs=num_envs,
            observation_space="state",
            randomness=randomness,
            max_env_steps=max_episode_steps,
            record=record,
            pos_scalar=1,
            rot_scalar=1,
            stiffness=1_000,
            damping=200,
        )
        env = FurnitureRLSimEnvMultiStepWrapper(
            env,
            n_obs_steps=obs_steps,
            n_action_steps=act_steps,
            prev_action=False,
            reset_within_step=False,
            pass_full_observations=False,
            normalization_path=normalization_path,
            sparse_reward=sparse_reward,
        )
        return env

    # avoid import error due incompatible gym versions
    from gym import spaces
    from env.gym_utils.async_vector_env import AsyncVectorEnv
    from env.gym_utils.sync_vector_env import SyncVectorEnv
    from env.gym_utils.wrapper import wrapper_dict

    __all__ = [
        "AsyncVectorEnv",
        "SyncVectorEnv",
        "VectorEnv",
        "VectorEnvWrapper",
        "make",
    ]

    # import the envs
    if robomimic_env_cfg_path is not None:
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
    elif "avoiding" in id:
        import gym_avoiding
    else:
        import d4rl.gym_mujoco
    from gym.envs import make as make_

    def _make_env():
        if robomimic_env_cfg_path is not None:
            obs_modality_dict = {
                "low_dim": (
                    wrappers.robomimic_image.low_dim_keys
                    if "robomimic_image" in wrappers
                    else wrappers.robomimic_lowdim.low_dim_keys
                ),
                "rgb": (
                    wrappers.robomimic_image.image_keys
                    if "robomimic_image" in wrappers
                    else None
                ),
            }
            if obs_modality_dict["rgb"] is None:
                obs_modality_dict.pop("rgb")
            ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
            if render_offscreen or use_image_obs:
                os.environ["MUJOCO_GL"] = "egl"
            with open(robomimic_env_cfg_path, "r") as f:
                env_meta = json.load(f)
            env_meta["reward_shaping"] = reward_shaping
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=render,
                # only way to not show collision geometry is to enable render_offscreen, which uses a lot of RAM.
                render_offscreen=render_offscreen,
                use_image_obs=use_image_obs,
                # render_gpu_device_id=0,
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            env.env.hard_reset = False
        else:  # d3il, gym
            if "kitchen" not in id:  # d4rl kitchen does not support rendering!
                kwargs["render"] = render
            env = make_(id, **kwargs)

        # add wrappers
        if wrappers is not None:
            for wrapper, args in wrappers.items():
                env = wrapper_dict[wrapper](env, **args)
        return env

    def dummy_env_fn():
        """TODO(allenzren): does this dummy env allow camera obs for other envs besides robomimic?"""
        import gym
        import numpy as np
        from env.gym_utils.wrapper.multi_step import MultiStep

        # Avoid importing or using env in the main process
        # to prevent OpenGL context issue with fork.
        # Create a fake env whose sole purpose is to provide
        # obs/action spaces and metadata.
        env = gym.Env()
        observation_space = spaces.Dict()
        if shape_meta is not None:  # rn only for images
            for key, value in shape_meta["obs"].items():
                shape = value["shape"]
                if key.endswith("rgb"):
                    min_value, max_value = -1, 1
                elif key.endswith("state"):
                    min_value, max_value = -1, 1
                else:
                    raise RuntimeError(f"Unsupported type {key}")
                observation_space[key] = spaces.Box(
                    low=min_value,
                    high=max_value,
                    shape=shape,
                    dtype=np.float32,
                )
        else:
            observation_space["state"] = gym.spaces.Box(
                -1,
                1,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        env.observation_space = observation_space
        env.action_space = gym.spaces.Box(-1, 1, shape=(action_dim,), dtype=np.int64)
        env.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": 12,
        }
        return MultiStep(env=env, n_obs_steps=wrappers.multi_step.n_obs_steps)

    env_fns = [_make_env for _ in range(num_envs)]
    return (
        AsyncVectorEnv(
            env_fns,
            dummy_env_fn=(
                dummy_env_fn if render or render_offscreen or use_image_obs else None
            ),
            delay_init="avoiding" in id,  # add delay for D3IL initialization
        )
        if asynchronous
        else SyncVectorEnv(env_fns)
    )
