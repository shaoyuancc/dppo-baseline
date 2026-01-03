from .multi_step import MultiStep
from .full_trajectory_multi_step import FullTrajectoryMultiStep
from .robomimic_lowdim import RobomimicLowdimWrapper
from .robomimic_image import RobomimicImageWrapper
from .d3il_lowdim import D3ilLowdimWrapper
from .mujoco_locomotion_lowdim import MujocoLocomotionLowdimWrapper


wrapper_dict = {
    "multi_step": MultiStep,
    "full_trajectory_multi_step": FullTrajectoryMultiStep,
    "robomimic_lowdim": RobomimicLowdimWrapper,
    "robomimic_image": RobomimicImageWrapper,
    "d3il_lowdim": D3ilLowdimWrapper,
    "mujoco_locomotion_lowdim": MujocoLocomotionLowdimWrapper,
}
