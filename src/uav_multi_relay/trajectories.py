"""Waypoint-following utilities for non-relay UAVs."""

from __future__ import annotations

import numpy as np

from .core import MotionLimits, UAVState
from .kinematics import make_velocity_feasible


class WaypointFollower:
    """Generate feasible velocities towards a cyclic or finite waypoint sequence."""

    def __init__(
        self,
        waypoints_m: object,
        *,
        cyclic: bool = True,
        arrival_tolerance_m: float = 2.0,
    ) -> None:
        waypoints = np.asarray(waypoints_m, dtype=float)
        if (
            waypoints.ndim != 2
            or waypoints.shape[0] < 1
            or waypoints.shape[1:] != (3,)
            or not np.all(np.isfinite(waypoints))
        ):
            raise ValueError("waypoints_m must be a finite array with shape (N, 3), N >= 1")
        if not np.isfinite(arrival_tolerance_m) or arrival_tolerance_m <= 0:
            raise ValueError("arrival_tolerance_m must be a positive finite value")
        self._waypoints_m = waypoints.copy()
        self._waypoints_m.setflags(write=False)
        self._cyclic = bool(cyclic)
        self._arrival_tolerance_m = float(arrival_tolerance_m)
        self._index = 0

    def reset(self) -> None:
        """Return the follower to its first waypoint."""
        self._index = 0

    def velocity_for(
        self,
        state: UAVState,
        limits: MotionLimits,
        delta_t_s: float,
    ) -> np.ndarray:
        """Return an acceleration- and speed-feasible velocity towards the target."""
        self._advance_reached_waypoints(state.position_m)
        displacement = self._waypoints_m[self._index] - state.position_m
        requested = np.zeros(3, dtype=float)
        if float(np.linalg.norm(displacement)) > self._arrival_tolerance_m:
            horizontal_distance = float(np.linalg.norm(displacement[:2]))
            if horizontal_distance > 0.0:
                requested[:2] = (
                    displacement[:2]
                    / horizontal_distance
                    * limits.max_horizontal_speed_mps
                )
            if displacement[2] != 0.0:
                requested[2] = (
                    limits.max_climb_speed_mps
                    if displacement[2] > 0
                    else -limits.max_descent_speed_mps
                )
        return make_velocity_feasible(requested, state.velocity_mps, limits, delta_t_s)

    def _advance_reached_waypoints(self, position_m: np.ndarray) -> None:
        for _ in range(len(self._waypoints_m)):
            if np.linalg.norm(self._waypoints_m[self._index] - position_m) > self._arrival_tolerance_m:
                return
            if self._index + 1 < len(self._waypoints_m):
                self._index += 1
            elif self._cyclic:
                self._index = 0
            else:
                return
            if len(self._waypoints_m) == 1:
                return
