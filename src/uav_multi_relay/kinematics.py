"""Velocity feasibility and state propagation functions."""

from __future__ import annotations

import numpy as np

from .core import MotionLimits, UAVState, _vector3


def _positive_finite(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return float(value)


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= maximum else vector * (maximum / norm)


def make_velocity_feasible(
    requested_velocity_mps: object,
    current_velocity_mps: object,
    limits: MotionLimits,
    delta_t_s: float,
) -> np.ndarray:
    """Apply acceleration and speed limits to a requested three-dimensional velocity."""
    requested = _vector3(requested_velocity_mps, "requested_velocity_mps")
    current = _vector3(current_velocity_mps, "current_velocity_mps")
    delta_t = _positive_finite(delta_t_s, "delta_t_s")

    horizontal_delta = requested[:2] - current[:2]
    horizontal_delta = _clip_norm(
        horizontal_delta, limits.max_horizontal_accel_mps2 * delta_t
    )
    horizontal_velocity = current[:2] + horizontal_delta

    vertical_delta = float(requested[2] - current[2])
    vertical_delta = float(
        np.clip(
            vertical_delta,
            -limits.max_vertical_accel_mps2 * delta_t,
            limits.max_vertical_accel_mps2 * delta_t,
        )
    )
    vertical_velocity = current[2] + vertical_delta

    horizontal_velocity = _clip_norm(
        horizontal_velocity, limits.max_horizontal_speed_mps
    )
    vertical_velocity = float(
        np.clip(
            vertical_velocity,
            -limits.max_descent_speed_mps,
            limits.max_climb_speed_mps,
        )
    )
    return np.array([horizontal_velocity[0], horizontal_velocity[1], vertical_velocity])


def advance_state(
    state: UAVState,
    applied_velocity_mps: object,
    delta_t_s: float,
) -> UAVState:
    """Advance a state using an already-feasible applied velocity."""
    applied_velocity = _vector3(applied_velocity_mps, "applied_velocity_mps")
    delta_t = _positive_finite(delta_t_s, "delta_t_s")
    return state.moved(state.position_m + applied_velocity * delta_t, applied_velocity)
