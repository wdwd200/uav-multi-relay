"""Immutable data types for UAV state and motion constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _vector3(value: object, name: str) -> np.ndarray:
    """Return an independent, finite three-dimensional float vector."""
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite floating-point vector with shape (3,)")
    result = vector.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class UAVState:
    """Position and velocity snapshot of a named UAV."""

    name: str
    position_m: np.ndarray
    velocity_mps: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _vector3(self.position_m, "position_m"))
        object.__setattr__(self, "velocity_mps", _vector3(self.velocity_mps, "velocity_mps"))

    def moved(self, position_m: object, velocity_mps: object) -> UAVState:
        """Create a new state while retaining this state's name."""
        return UAVState(self.name, position_m, velocity_mps)


@dataclass(frozen=True)
class MotionLimits:
    """Directional speed and acceleration limits for a UAV."""

    max_horizontal_speed_mps: float
    max_climb_speed_mps: float
    max_descent_speed_mps: float
    max_horizontal_accel_mps2: float
    max_vertical_accel_mps2: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite value")
