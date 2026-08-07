"""Velocity feasibility and state propagation functions."""

from __future__ import annotations

import numpy as np

from .core import MotionLimits, UAVState, _vector3


def _positive_finite(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return float(value)


def _speed_limit_tolerance(limit: float) -> float:
    """Return a machine-precision-scale tolerance for one speed limit."""
    return float(64.0 * np.finfo(float).eps * max(1.0, abs(limit)))


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum:
        return vector.copy()
    clipped = vector * (maximum / norm)
    # A second scale uses the next representable value below the limit so
    # rounding cannot leave the returned norm just above the configured cap.
    for _ in range(2):
        clipped_norm = float(np.linalg.norm(clipped))
        if clipped_norm <= maximum:
            break
        clipped *= np.nextafter(maximum, 0.0) / clipped_norm
    return clipped


def make_velocity_feasible(
    requested_velocity_mps: object,
    current_velocity_mps: object,
    limits: MotionLimits,
    delta_t_s: float,
) -> np.ndarray:
    """Apply acceleration and speed limits to a requested three-dimensional velocity."""
    requested = _vector3(requested_velocity_mps, "requested_velocity_mps")
    current = _vector3(current_velocity_mps, "current_velocity_mps").copy()
    delta_t = _positive_finite(delta_t_s, "delta_t_s")
    horizontal_tolerance = _speed_limit_tolerance(limits.max_horizontal_speed_mps)
    climb_tolerance = _speed_limit_tolerance(limits.max_climb_speed_mps)
    descent_tolerance = _speed_limit_tolerance(limits.max_descent_speed_mps)
    horizontal_norm = float(np.linalg.norm(current[:2]))
    if horizontal_norm > limits.max_horizontal_speed_mps + horizontal_tolerance:
        raise ValueError("current_velocity_mps must satisfy the configured speed limits")
    if current[2] > limits.max_climb_speed_mps + climb_tolerance:
        raise ValueError("current_velocity_mps must satisfy the configured speed limits")
    if current[2] < -limits.max_descent_speed_mps - descent_tolerance:
        raise ValueError("current_velocity_mps must satisfy the configured speed limits")

    # Normalize values accepted only because they are within floating-point
    # tolerance before applying acceleration limits or propagating state.
    if horizontal_norm > limits.max_horizontal_speed_mps:
        current[:2] = _clip_norm(current[:2], limits.max_horizontal_speed_mps)
    current[2] = float(
        np.clip(current[2], -limits.max_descent_speed_mps, limits.max_climb_speed_mps)
    )

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
