"""Action mapping and pre-execution safety filtering for relay motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FlightBounds
from .core import MotionLimits, UAVState, _vector3
from .kinematics import make_velocity_feasible


def _copy_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SafetyFilterResult:
    """Requested and safely applicable relay velocities for one simulation step."""

    requested_velocities_mps: np.ndarray
    applied_velocities_mps: np.ndarray
    intervention_norms: np.ndarray
    scale: float

    def __post_init__(self) -> None:
        requested = np.asarray(self.requested_velocities_mps, dtype=float)
        applied = np.asarray(self.applied_velocities_mps, dtype=float)
        if requested.ndim != 2 or requested.shape[1:] != (3,):
            raise ValueError("requested_velocities_mps must have shape (K, 3)")
        object.__setattr__(
            self,
            "requested_velocities_mps",
            _copy_array(requested, requested.shape, "requested_velocities_mps"),
        )
        object.__setattr__(
            self,
            "applied_velocities_mps",
            _copy_array(applied, requested.shape, "applied_velocities_mps"),
        )
        object.__setattr__(
            self,
            "intervention_norms",
            _copy_array(self.intervention_norms, (requested.shape[0],), "intervention_norms"),
        )
        if not np.isfinite(self.scale) or not 0.0 <= self.scale <= 1.0:
            raise ValueError("scale must be finite and within [0, 1]")


class NoFeasibleActionError(RuntimeError):
    """Raised when no relay velocity interpolation satisfies all hard constraints."""


def normalized_action_to_velocity(action: object, limits: MotionLimits) -> np.ndarray:
    """Map one normalized relay action to a requested physical velocity."""
    normalized = _vector3(action, "action")
    if np.any(normalized < -1.0) or np.any(normalized > 1.0):
        raise ValueError("action components must be within [-1, 1]")
    horizontal = normalized[:2].copy()
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm > 1.0:
        horizontal /= horizontal_norm
    horizontal *= limits.max_horizontal_speed_mps
    vertical = (
        normalized[2] * limits.max_climb_speed_mps
        if normalized[2] >= 0.0
        else normalized[2] * limits.max_descent_speed_mps
    )
    return np.array([horizontal[0], horizontal[1], vertical], dtype=float)


def velocity_to_normalized_action(
    velocity_mps: object, limits: MotionLimits
) -> np.ndarray:
    """Map one feasible physical relay velocity back to a normalized action."""
    velocity = _vector3(velocity_mps, "velocity_mps")
    horizontal_limit = limits.max_horizontal_speed_mps
    horizontal_norm = float(np.linalg.norm(velocity[:2]))
    horizontal_tolerance = 1e-9 * max(1.0, horizontal_limit)
    climb_tolerance = 1e-9 * max(1.0, limits.max_climb_speed_mps)
    descent_tolerance = 1e-9 * max(1.0, limits.max_descent_speed_mps)
    if horizontal_norm > horizontal_limit + horizontal_tolerance:
        raise ValueError("horizontal velocity exceeds the configured limit")
    if velocity[2] > limits.max_climb_speed_mps + climb_tolerance:
        raise ValueError("climb velocity exceeds the configured limit")
    if velocity[2] < -limits.max_descent_speed_mps - descent_tolerance:
        raise ValueError("descent velocity exceeds the configured limit")

    horizontal = velocity[:2] / horizontal_limit
    horizontal_action_norm = float(np.linalg.norm(horizontal))
    if horizontal_action_norm > 1.0:
        horizontal *= 1.0 / horizontal_action_norm
    vertical = (
        velocity[2] / limits.max_climb_speed_mps
        if velocity[2] >= 0.0
        else velocity[2] / limits.max_descent_speed_mps
    )
    result = np.array([horizontal[0], horizontal[1], vertical], dtype=float)
    return np.clip(result, -1.0, 1.0)


def filter_relay_velocities(
    relay_states: tuple[UAVState, ...],
    requested_velocities_mps: object,
    limits: MotionLimits,
    delta_t_s: float,
    flight_bounds: FlightBounds,
    hard_safety_distance_m: float,
    hard_max_link_distance_m: float,
    high_candidate_position_m: object,
    low_candidate_position_m: object,
) -> SafetyFilterResult:
    """Find the largest feasible interpolation from current to requested relay velocity."""
    if not relay_states:
        raise ValueError("at least one relay state is required")
    requested = _copy_array(
        requested_velocities_mps, (len(relay_states), 3), "requested_velocities_mps"
    )
    if not np.isfinite(delta_t_s) or delta_t_s <= 0:
        raise ValueError("delta_t_s must be a positive finite value")
    if not np.isfinite(hard_safety_distance_m) or hard_safety_distance_m <= 0:
        raise ValueError("hard_safety_distance_m must be a positive finite value")
    if not np.isfinite(hard_max_link_distance_m) or hard_max_link_distance_m <= 0:
        raise ValueError("hard_max_link_distance_m must be a positive finite value")

    current = np.stack([state.velocity_mps for state in relay_states])
    feasible = np.stack(
        [
            make_velocity_feasible(request, state.velocity_mps, limits, delta_t_s)
            for request, state in zip(requested, relay_states)
        ]
    )
    high_position = _vector3(high_candidate_position_m, "high_candidate_position_m")
    low_position = _vector3(low_candidate_position_m, "low_candidate_position_m")

    for scale in np.linspace(1.0, 0.0, 21):
        applied = current + scale * (feasible - current)
        relay_positions = np.stack(
            [state.position_m + velocity * delta_t_s for state, velocity in zip(relay_states, applied)]
        )
        if _positions_are_safe(
            high_position,
            relay_positions,
            low_position,
            flight_bounds,
            hard_safety_distance_m,
            hard_max_link_distance_m,
        ):
            return SafetyFilterResult(
                requested,
                applied,
                np.linalg.norm(requested - applied, axis=1),
                float(scale),
            )
    raise NoFeasibleActionError("no interpolated relay velocity satisfies the hard constraints")


def _positions_are_safe(
    high_position: np.ndarray,
    relay_positions: np.ndarray,
    low_position: np.ndarray,
    flight_bounds: FlightBounds,
    hard_safety_distance_m: float,
    hard_max_link_distance_m: float,
) -> bool:
    if not all(flight_bounds.contains(position) for position in relay_positions):
        return False
    positions = np.vstack((high_position, relay_positions, low_position))
    pairwise = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(pairwise, axis=2)
    np.fill_diagonal(distances, np.inf)
    if float(np.min(distances)) < hard_safety_distance_m:
        return False
    adjacent_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return bool(np.all(adjacent_distances <= hard_max_link_distance_m))
