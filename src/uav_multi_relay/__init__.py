"""Multi-relay UAV communication research project."""

from .core import MotionLimits, UAVState
from .config import RewardWeights, default_environment_config, scenario_environment_config
from .environment import MultiRelayEnvironment

__version__ = "0.1.0"

__all__ = [
    "MotionLimits",
    "RewardWeights",
    "MultiRelayEnvironment",
    "UAVState",
    "default_environment_config",
    "scenario_environment_config",
]
