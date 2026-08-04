"""A synchronous, dependency-free multi-relay UAV environment."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .communication import (
    channel_power_gain,
    compute_link_geometry,
    dipole_gain,
    optimal_tdma_rate,
    shannon_capacity_bps,
    snr_linear,
)
from .config import (
    EndpointTrajectoryConfig,
    EnvironmentConfig,
    default_environment_config,
)
from .core import UAVState
from .kinematics import advance_state
from .safety import (
    NoFeasibleActionError,
    SafetyFilterResult,
    filter_relay_velocities,
    normalized_action_to_velocity,
)
from .trajectories import WaypointFollower


class MultiRelayEnvironment:
    """Simulate endpoint trajectories and jointly controlled relay UAVs."""

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.config = default_environment_config() if config is None else config
        if not isinstance(self.config, EnvironmentConfig):
            raise ValueError("config must be an EnvironmentConfig instance")
        self._high_follower = WaypointFollower(
            np.zeros((1, 3))
        )
        self._low_follower = WaypointFollower(
            np.zeros((1, 3))
        )
        self._states: tuple[UAVState, ...] = ()
        self._last_applied_relay_velocities = np.zeros((self.config.num_relays, 3))
        self._step_index = 0
        self._terminated = False
        self._truncated = False
        self._rng = np.random.default_rng()

    @property
    def states(self) -> tuple[UAVState, ...]:
        """Return ordered H, relays, L state snapshots."""
        self._require_reset()
        return self._states

    @property
    def high_state(self) -> UAVState:
        """Return the current high endpoint state."""
        return self.states[0]

    @property
    def relay_states(self) -> tuple[UAVState, ...]:
        """Return the current relay state snapshots."""
        return self.states[1:-1]

    @property
    def low_state(self) -> UAVState:
        """Return the current low endpoint state."""
        return self.states[-1]

    @property
    def step_index(self) -> int:
        """Return the number of successfully committed steps."""
        return self._step_index

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        """Initialize the fixed, valid H-to-L relay chain."""
        self._rng = np.random.default_rng(seed)
        self._high_follower.reset()
        self._low_follower.reset()
        high_position, relay_positions, low_position = self._sample_initial_chain()
        self._high_follower = WaypointFollower(
            self._make_endpoint_waypoints(high_position, self.config.high_trajectory),
            arrival_tolerance_m=self.config.high_trajectory.arrival_tolerance_m,
        )
        self._low_follower = WaypointFollower(
            self._make_endpoint_waypoints(low_position, self.config.low_trajectory),
            arrival_tolerance_m=self.config.low_trajectory.arrival_tolerance_m,
        )
        self._states = (
            UAVState("H", high_position, np.zeros(3)),
            *tuple(
                UAVState(f"R{index + 1}", position, np.zeros(3))
                for index, position in enumerate(relay_positions)
            ),
            UAVState("L", low_position, np.zeros(3)),
        )
        self._last_applied_relay_velocities = np.zeros((self.config.num_relays, 3))
        self._step_index = 0
        self._terminated = False
        self._truncated = False
        communication = self._communication(self._states, self._states)
        reward_terms = self._zero_reward_terms()
        return (
            self._observation(communication["capacities_bps"]),
            self._info(
                self._states,
                np.zeros((self.config.num_relays, 3)),
                self._last_applied_relay_velocities,
                np.zeros(self.config.num_relays),
                1.0,
                communication,
                reward_terms,
            ),
        )

    def step(
        self, actions: object
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        """Synchronously advance all UAVs by one environment time step."""
        self._require_reset()
        if self._terminated or self._truncated:
            raise RuntimeError("reset() is required before stepping a completed episode")
        normalized_actions = self._validate_actions(actions)
        old_states = self._states
        high_velocity = self._high_follower.velocity_for(
            old_states[0], self.config.high_motion_limits, self.config.delta_t_s
        )
        low_velocity = self._low_follower.velocity_for(
            old_states[-1], self.config.low_motion_limits, self.config.delta_t_s
        )
        requested = np.stack(
            [
                normalized_action_to_velocity(action, self.config.relay_motion_limits)
                for action in normalized_actions
            ]
        )
        high_next = advance_state(old_states[0], high_velocity, self.config.delta_t_s)
        low_next = advance_state(old_states[-1], low_velocity, self.config.delta_t_s)

        if not self.config.flight_bounds.contains(high_next.position_m):
            return self._terminate_for_candidate(
                old_states,
                requested,
                "high endpoint waypoint candidate is outside flight bounds",
            )
        if not self.config.flight_bounds.contains(low_next.position_m):
            return self._terminate_for_candidate(
                old_states,
                requested,
                "low endpoint waypoint candidate is outside flight bounds",
            )

        try:
            filtered = filter_relay_velocities(
                old_states[1:-1],
                requested,
                self.config.relay_motion_limits,
                self.config.delta_t_s,
                self.config.flight_bounds,
                self.config.hard_safety_distance_m,
                self.config.hard_max_link_distance_m,
                high_next.position_m,
                low_next.position_m,
            )
        except NoFeasibleActionError as error:
            return self._terminate_for_candidate(old_states, requested, str(error))

        relay_next = tuple(
            advance_state(state, velocity, self.config.delta_t_s)
            for state, velocity in zip(old_states[1:-1], filtered.applied_velocities_mps)
        )
        next_states = (high_next, *relay_next, low_next)
        self._states = next_states
        self._last_applied_relay_velocities = filtered.applied_velocities_mps.copy()
        self._step_index += 1
        self._truncated = self._step_index >= self.config.max_steps
        communication = self._communication(old_states, next_states)
        reward, reward_terms = self._reward(old_states, next_states, filtered, communication)
        info = self._info(
            next_states,
            filtered.requested_velocities_mps,
            filtered.applied_velocities_mps,
            filtered.intervention_norms,
            filtered.scale,
            communication,
            reward_terms,
        )
        return (
            self._observation(communication["capacities_bps"]),
            reward,
            False,
            self._truncated,
            info,
        )

    def _validate_actions(self, actions: object) -> np.ndarray:
        array = np.asarray(actions, dtype=float)
        if array.shape != (self.config.num_relays, 3) or not np.all(np.isfinite(array)):
            raise ValueError(f"actions must be finite with shape ({self.config.num_relays}, 3)")
        if np.any(array < -1.0) or np.any(array > 1.0):
            raise ValueError("actions must be within [-1, 1]")
        return array.copy()

    def _sample_initial_chain(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct a hard-feasible H-to-L line segment and equally spaced relays."""
        bounds = self.config.flight_bounds
        high_trajectory = self.config.high_trajectory
        low_trajectory = self.config.low_trajectory
        waypoint_radius = max(
            high_trajectory.waypoint_radius_m,
            low_trajectory.waypoint_radius_m,
        )
        endpoint_turn_margin = max(
            limits.max_horizontal_speed_mps**2
            / (2.0 * limits.max_horizontal_accel_mps2)
            + limits.max_horizontal_speed_mps * self.config.delta_t_s
            for limits in (
                self.config.high_motion_limits,
                self.config.low_motion_limits,
            )
        ) + max(
            high_trajectory.arrival_tolerance_m,
            low_trajectory.arrival_tolerance_m,
        )
        inset = waypoint_radius + endpoint_turn_margin
        horizontal_lower = bounds.minimum_m[:2] + inset
        horizontal_upper = bounds.maximum_m[:2] - inset
        if np.any(horizontal_lower > horizontal_upper):
            raise ValueError(
                "unable to construct a hard-feasible initial relay chain: "
                "flight bounds leave no room for configured endpoint waypoint radius"
            )

        segment_count = self.config.num_relays + 1
        total_minimum = segment_count * self.config.hard_safety_distance_m
        total_maximum = segment_count * self.config.hard_max_link_distance_m
        horizontal_span = horizontal_upper - horizontal_lower
        horizontal_diagonal = float(np.linalg.norm(horizontal_span))
        gap_minimum = high_trajectory.altitude_min_m - low_trajectory.altitude_max_m
        gap_maximum = high_trajectory.altitude_max_m - low_trajectory.altitude_min_m
        gap_lower = max(
            gap_minimum,
            float(np.sqrt(max(total_minimum**2 - horizontal_diagonal**2, 0.0))),
        )
        gap_upper = min(gap_maximum, total_maximum)
        if gap_lower > gap_upper:
            raise ValueError(
                "unable to construct a hard-feasible initial relay chain for the configured bounds"
            )
        altitude_gap = float(self._rng.uniform(gap_lower, gap_upper))
        low_altitude_lower = max(
            low_trajectory.altitude_min_m,
            high_trajectory.altitude_min_m - altitude_gap,
        )
        low_altitude_upper = min(
            low_trajectory.altitude_max_m,
            high_trajectory.altitude_max_m - altitude_gap,
        )
        if low_altitude_lower > low_altitude_upper:
            raise ValueError("unable to select endpoint altitudes for the configured trajectory ranges")
        low_altitude = float(self._rng.uniform(low_altitude_lower, low_altitude_upper))
        high_altitude = low_altitude + altitude_gap

        total_lower = max(total_minimum, altitude_gap)
        total_upper = min(
            total_maximum,
            float(np.sqrt(horizontal_diagonal**2 + altitude_gap**2)),
        )
        if total_lower > total_upper:
            raise ValueError(
                "unable to construct a hard-feasible initial relay chain for the configured bounds"
            )
        preferred_total = segment_count * (
            self.config.hard_safety_distance_m + self.config.hard_max_link_distance_m
        ) / 2.0
        total_distance = float(np.clip(preferred_total, total_lower, total_upper))
        horizontal_distance = float(
            np.sqrt(max(total_distance**2 - altitude_gap**2, 0.0))
        )
        horizontal_delta = self._horizontal_delta(horizontal_distance, horizontal_span)
        low_horizontal = self._sample_low_horizontal(
            horizontal_delta, horizontal_lower, horizontal_upper
        )
        high_horizontal = low_horizontal + horizontal_delta
        high = np.array([high_horizontal[0], high_horizontal[1], high_altitude])
        low = np.array([low_horizontal[0], low_horizontal[1], low_altitude])
        relay_positions = np.linspace(high, low, segment_count + 1)[1:-1]
        positions = np.vstack((high, relay_positions, low))
        if not all(bounds.contains(position) for position in positions):
            raise ValueError("constructed initial relay chain is outside flight bounds")
        adjacent = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        if (
            np.any(adjacent < self.config.hard_safety_distance_m - 1e-9)
            or np.any(adjacent > self.config.hard_max_link_distance_m + 1e-9)
        ):
            raise ValueError("constructed initial relay chain violates hard link constraints")
        return high, relay_positions, low

    def _horizontal_delta(self, distance: float, span: np.ndarray) -> np.ndarray:
        """Return a seeded signed 2D vector of the requested length that fits a box."""
        if distance == 0.0:
            return np.zeros(2)
        if distance <= span[0]:
            components = np.array([distance, 0.0])
        elif distance <= span[1]:
            components = np.array([0.0, distance])
        else:
            first = min(distance, span[0])
            second = float(np.sqrt(max(distance**2 - first**2, 0.0)))
            if second > span[1] + 1e-9:
                raise ValueError("configured flight bounds cannot contain the initial relay chain")
            components = np.array([first, second])
        signs = self._rng.choice(np.array([-1.0, 1.0]), size=2)
        return components * signs

    def _sample_low_horizontal(
        self, delta: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        low_lower = lower + np.maximum(-delta, 0.0)
        low_upper = upper - np.maximum(delta, 0.0)
        if np.any(low_lower > low_upper):
            raise ValueError("configured flight bounds cannot contain the initial relay chain")
        return self._rng.uniform(low_lower, low_upper)

    def _make_endpoint_waypoints(
        self, anchor: np.ndarray, trajectory: EndpointTrajectoryConfig
    ) -> np.ndarray:
        """Create a bounded seeded loop with at least one unreached waypoint."""
        bounds = self.config.flight_bounds
        waypoints = [anchor.copy()]
        phase = float(self._rng.uniform(0.0, 2.0 * np.pi))
        for index in range(1, trajectory.waypoint_count):
            angle = phase + 2.0 * np.pi * index / trajectory.waypoint_count
            horizontal_offset = trajectory.waypoint_radius_m * np.array(
                [np.cos(angle), np.sin(angle)]
            )
            altitude = float(
                self._rng.uniform(trajectory.altitude_min_m, trajectory.altitude_max_m)
            )
            waypoint = np.array(
                [
                    anchor[0] + horizontal_offset[0],
                    anchor[1] + horizontal_offset[1],
                    altitude,
                ]
            )
            if not bounds.contains(waypoint):
                raise ValueError("configured endpoint waypoint is outside flight bounds")
            waypoints.append(waypoint)
        return np.asarray(waypoints, dtype=float)

    def _terminate_for_candidate(
        self,
        old_states: tuple[UAVState, ...],
        requested: np.ndarray,
        reason: str,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        self._terminated = True
        communication = self._communication(old_states, old_states)
        applied = np.stack([state.velocity_mps for state in old_states[1:-1]])
        interventions = np.linalg.norm(requested - applied, axis=1)
        reward_terms = self._zero_reward_terms()
        reward_terms["failure_penalty"] = 1.0
        info = self._info(
            old_states,
            requested,
            applied,
            interventions,
            0.0,
            communication,
            reward_terms,
        )
        info["failure_reason"] = reason
        return self._observation(communication["capacities_bps"]), -1.0, True, False, info

    def _communication(
        self,
        old_states: tuple[UAVState, ...],
        new_states: tuple[UAVState, ...],
    ) -> dict[str, np.ndarray | float]:
        channel = self.config.channel
        midpoint_positions = [
            (old.position_m + new.position_m) / 2.0
            for old, new in zip(old_states, new_states)
        ]
        distances: list[float] = []
        elevations: list[float] = []
        capacities: list[float] = []
        for tx_position, rx_position in zip(midpoint_positions, midpoint_positions[1:]):
            geometry = compute_link_geometry(tx_position, rx_position)
            antenna_gain = dipole_gain(
                geometry.elevation_angle_rad,
                channel.maximum_antenna_gain_linear,
                channel.minimum_antenna_gain_linear,
            )
            gain = channel_power_gain(
                geometry.distance_3d_m,
                channel.reference_gain_linear,
                channel.reference_distance_m,
                channel.path_loss_exponent,
                antenna_gain,
                antenna_gain,
                channel.minimum_distance_m,
            )
            snr = snr_linear(
                channel.transmit_power_w,
                gain,
                channel.noise_psd_w_per_hz,
                channel.bandwidth_hz,
                channel.noise_figure_linear,
            )
            distances.append(geometry.distance_3d_m)
            elevations.append(geometry.elevation_angle_rad)
            capacities.append(shannon_capacity_bps(channel.bandwidth_hz, snr))
        capacities_array = np.asarray(capacities, dtype=float)
        rate, fractions = optimal_tdma_rate(capacities_array)
        return {
            "distances_m": np.asarray(distances, dtype=float),
            "elevation_angles_rad": np.asarray(elevations, dtype=float),
            "capacities_bps": capacities_array,
            "tdma_fractions": fractions,
            "rate_e2e_bps": rate,
        }

    def _observation(self, capacities_bps: np.ndarray) -> dict[str, np.ndarray]:
        positions = np.stack([state.position_m for state in self._states])
        velocities = np.stack([state.velocity_mps for state in self._states])
        span = self.config.flight_bounds.maximum_m - self.config.flight_bounds.minimum_m
        normalized_positions = 2.0 * (positions - self.config.flight_bounds.minimum_m) / span - 1.0
        velocity_scale = self._velocity_scale()
        normalized_velocities = velocities / velocity_scale
        progress = self._step_index / self.config.max_steps
        local_rows = []
        for relay_index in range(self.config.num_relays):
            node_index = relay_index + 1
            local_rows.append(
                np.concatenate(
                    (
                        normalized_positions[node_index],
                        normalized_velocities[node_index],
                        (positions[node_index - 1] - positions[node_index]) / span,
                        (velocities[node_index - 1] - velocities[node_index]) / velocity_scale,
                        (positions[node_index + 1] - positions[node_index]) / span,
                        (velocities[node_index + 1] - velocities[node_index]) / velocity_scale,
                        self._last_applied_relay_velocities[relay_index] / velocity_scale,
                        np.array(
                            [
                                relay_index / max(self.config.num_relays - 1, 1),
                                progress,
                            ]
                        ),
                    )
                )
            )
        global_state = np.concatenate(
            (
                normalized_positions.ravel(),
                normalized_velocities.ravel(),
                capacities_bps / self.config.rate_reference_bps,
                np.array([progress]),
            )
        )
        return {"local": np.asarray(local_rows, dtype=float), "global": global_state.astype(float)}

    def _reward(
        self,
        old_states: tuple[UAVState, ...],
        new_states: tuple[UAVState, ...],
        filtered: SafetyFilterResult,
        communication: dict[str, np.ndarray | float],
    ) -> tuple[float, dict[str, float]]:
        distances = np.asarray(communication["distances_m"], dtype=float)
        positions = np.stack([state.position_m for state in new_states])
        pair_distances = [
            float(np.linalg.norm(positions[first] - positions[second]))
            for first, second in combinations(range(len(new_states)), 2)
        ]
        link_margin = 0.8 * self.config.hard_max_link_distance_m
        link_cost = float(
            np.mean(
                np.maximum(
                    0.0,
                    (distances - link_margin)
                    / (self.config.hard_max_link_distance_m - link_margin),
                )
            )
        )
        separation_cost = float(
            np.mean(
                [
                    max(0.0, (self.config.soft_safety_distance_m - distance) / self.config.soft_safety_distance_m)
                    for distance in pair_distances
                ]
            )
        )
        intervention_cost = float(
            np.mean(filtered.intervention_norms)
            / self.config.relay_motion_limits.max_horizontal_speed_mps
        )
        velocities = np.stack([state.velocity_mps for state in new_states])
        old_velocities = np.stack([state.velocity_mps for state in old_states])
        speed_scale = max(
            self.config.relay_motion_limits.max_horizontal_speed_mps,
            self.config.high_motion_limits.max_horizontal_speed_mps,
            self.config.low_motion_limits.max_horizontal_speed_mps,
        )
        motion_cost = float(
            np.mean(np.sum((velocities / speed_scale) ** 2, axis=1))
            + np.mean(np.sum(((velocities - old_velocities) / speed_scale) ** 2, axis=1))
        )
        rate_reward = float(communication["rate_e2e_bps"]) / self.config.rate_reference_bps
        terms = {
            "rate_reward": rate_reward,
            "link_cost": link_cost,
            "separation_cost": separation_cost,
            "intervention_cost": intervention_cost,
            "motion_cost": motion_cost,
        }
        return float(rate_reward - link_cost - separation_cost - intervention_cost - motion_cost), terms

    def _info(
        self,
        states: tuple[UAVState, ...],
        requested: np.ndarray,
        applied: np.ndarray,
        interventions: np.ndarray,
        scale: float,
        communication: dict[str, np.ndarray | float],
        reward_terms: dict[str, float],
    ) -> dict[str, object]:
        positions = np.stack([state.position_m for state in states])
        velocities = np.stack([state.velocity_mps for state in states])
        minimum_distance = min(
            float(np.linalg.norm(positions[first] - positions[second]))
            for first, second in combinations(range(len(states)), 2)
        )
        return {
            "positions_m": positions.copy(),
            "velocities_mps": velocities.copy(),
            "requested_relay_velocities_mps": requested.copy(),
            "applied_relay_velocities_mps": applied.copy(),
            "intervention_norms": interventions.copy(),
            "safety_scale": float(scale),
            "hop_distances_m": np.asarray(communication["distances_m"], dtype=float).copy(),
            "hop_elevation_angles_rad": np.asarray(communication["elevation_angles_rad"], dtype=float).copy(),
            "hop_capacities_bps": np.asarray(communication["capacities_bps"], dtype=float).copy(),
            "tdma_fractions": np.asarray(communication["tdma_fractions"], dtype=float).copy(),
            "rate_e2e_bps": float(communication["rate_e2e_bps"]),
            "minimum_uav_distance_m": minimum_distance,
            "reward_terms": dict(reward_terms),
            "step_index": self._step_index,
        }

    def _velocity_scale(self) -> np.ndarray:
        vertical = max(
            self.config.relay_motion_limits.max_climb_speed_mps,
            self.config.relay_motion_limits.max_descent_speed_mps,
            self.config.high_motion_limits.max_climb_speed_mps,
            self.config.high_motion_limits.max_descent_speed_mps,
            self.config.low_motion_limits.max_climb_speed_mps,
            self.config.low_motion_limits.max_descent_speed_mps,
        )
        horizontal = max(
            self.config.relay_motion_limits.max_horizontal_speed_mps,
            self.config.high_motion_limits.max_horizontal_speed_mps,
            self.config.low_motion_limits.max_horizontal_speed_mps,
        )
        return np.array([horizontal, horizontal, vertical])

    @staticmethod
    def _zero_reward_terms() -> dict[str, float]:
        return {
            "rate_reward": 0.0,
            "link_cost": 0.0,
            "separation_cost": 0.0,
            "intervention_cost": 0.0,
            "motion_cost": 0.0,
        }

    def _require_reset(self) -> None:
        if not self._states:
            raise RuntimeError("reset() must be called before accessing environment state")
