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
from .config import EnvironmentConfig, default_environment_config
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
            np.array(
                [
                    [-300.0, 0.0, 150.0],
                    [-300.0, 80.0, 160.0],
                    [-260.0, 80.0, 150.0],
                    [-260.0, 0.0, 150.0],
                ]
            )
        )
        self._low_follower = WaypointFollower(
            np.array(
                [
                    [300.0, 0.0, 150.0],
                    [300.0, -80.0, 140.0],
                    [260.0, -80.0, 150.0],
                    [260.0, 0.0, 150.0],
                ]
            )
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
        high_position = np.array([-300.0, 0.0, 150.0])
        low_position = np.array([300.0, 0.0, 150.0])
        relay_positions = np.linspace(
            high_position, low_position, self.config.num_relays + 2
        )[1:-1]
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
            info["failure_reason"] = str(error)
            return self._observation(communication["capacities_bps"]), -1.0, True, False, info

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
