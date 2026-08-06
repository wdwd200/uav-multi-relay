"""Short, reproducible diagnostics for reward and scenario sensitivity."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np

from ..baselines import equal_spacing_actions, stationary_actions
from ..config import EnvironmentConfig, EndpointTrajectoryConfig
from ..environment import MultiRelayEnvironment


@dataclass(frozen=True)
class ScenarioDiagnosticConfig:
    waypoint_radii_m: tuple[float, ...] = (30.0, 60.0, 90.0, 120.0)
    max_steps_values: tuple[int, ...] = (100, 250)
    episodes: int = 5
    seed: int = 30_000
    policies: tuple[str, ...] = ("stationary", "equal_spacing")

    def __post_init__(self) -> None:
        if not isinstance(self.waypoint_radii_m, tuple) or not self.waypoint_radii_m:
            raise ValueError("waypoint_radii_m must be a non-empty tuple")
        for value in self.waypoint_radii_m:
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0.0:
                raise ValueError("waypoint radii must be positive finite values")
        if not isinstance(self.max_steps_values, tuple) or not self.max_steps_values:
            raise ValueError("max_steps_values must be a non-empty tuple")
        for value in self.max_steps_values:
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError("max_steps_values must contain positive integers")
        if isinstance(self.episodes, bool) or not isinstance(self.episodes, Integral) or self.episodes <= 0:
            raise ValueError("episodes must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("seed must be an integer")
        if not isinstance(self.policies, tuple) or not self.policies or any(policy not in ("stationary", "equal_spacing") for policy in self.policies) or len(set(self.policies)) != len(self.policies):
            raise ValueError("policies must be distinct stationary/equal_spacing names")
        object.__setattr__(self, "waypoint_radii_m", tuple(float(value) for value in self.waypoint_radii_m))
        object.__setattr__(self, "max_steps_values", tuple(int(value) for value in self.max_steps_values))
        object.__setattr__(self, "episodes", int(self.episodes))


@dataclass(frozen=True)
class ScenarioEpisodeDiagnostic:
    waypoint_radius_m: float
    max_steps: int
    policy: str
    episode_index: int
    episode_seed: int
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    failure_reason: str | None
    mean_rate_e2e_bps: float
    min_rate_e2e_bps: float
    mean_rate_reward: float
    mean_link_cost: float
    mean_separation_cost: float
    mean_intervention_cost: float
    mean_motion_cost: float
    intervention_rate: float
    mean_high_displacement_m: float
    max_high_displacement_m: float
    mean_low_displacement_m: float
    max_low_displacement_m: float
    relay_path_length_m: float
    mean_min_hop_capacity_bps: float
    mean_max_hop_distance_m: float


@dataclass(frozen=True)
class ScenarioDiagnosticSummary:
    waypoint_radius_m: float
    max_steps: int
    policy: str
    completed_episodes: int
    termination_rate: float
    mean_return: float
    return_std: float
    mean_rate_e2e_bps: float
    minimum_rate_e2e_bps: float
    mean_rate_reward: float
    mean_link_cost: float
    mean_separation_cost: float
    mean_intervention_cost: float
    mean_motion_cost: float
    mean_intervention_rate: float
    mean_high_displacement_m: float
    mean_low_displacement_m: float
    mean_relay_path_length_m: float
    mean_min_hop_capacity_bps: float
    mean_max_hop_distance_m: float
    mean_episode_length: float


@dataclass(frozen=True)
class ScenarioDiagnosticResult:
    config: ScenarioDiagnosticConfig
    episode_results: tuple[ScenarioEpisodeDiagnostic, ...]
    summaries: tuple[ScenarioDiagnosticSummary, ...]


def _scenario_config(base: EnvironmentConfig, radius: float, max_steps: int) -> EnvironmentConfig:
    high = EndpointTrajectoryConfig(base.high_trajectory.altitude_min_m, base.high_trajectory.altitude_max_m, radius, base.high_trajectory.waypoint_count, base.high_trajectory.arrival_tolerance_m)
    low = EndpointTrajectoryConfig(base.low_trajectory.altitude_min_m, base.low_trajectory.altitude_max_m, radius, base.low_trajectory.waypoint_count, base.low_trajectory.arrival_tolerance_m)
    return base.__class__(
        base.num_relays, base.delta_t_s, max_steps, base.relay_motion_limits,
        base.high_motion_limits, base.low_motion_limits, base.flight_bounds,
        base.hard_safety_distance_m, base.soft_safety_distance_m,
        base.hard_max_link_distance_m, base.rate_reference_bps, base.channel,
        high, low, base.reward_weights,
    )


def _episode(env: MultiRelayEnvironment, policy: str, seed: int, radius: float, max_steps: int, episode_index: int) -> ScenarioEpisodeDiagnostic:
    observation, _ = env.reset(seed=seed)
    initial = np.asarray(observation["global"], dtype=float)
    initial_positions = np.asarray(env.states, dtype=object)
    initial_high = env.high_state.position_m.copy()
    initial_low = env.low_state.position_m.copy()
    relay_path = 0.0
    previous_relays = np.stack([state.position_m for state in env.relay_states]).copy()
    returns = 0.0
    rates: list[float] = []
    term_values: dict[str, list[float]] = {name: [] for name in ("rate_reward", "link_cost", "separation_cost", "intervention_cost", "motion_cost")}
    interventions = 0
    high_displacements: list[float] = []
    low_displacements: list[float] = []
    min_capacities: list[float] = []
    max_distances: list[float] = []
    failure_reason: str | None = None
    terminated = truncated = False
    length = 0
    while not (terminated or truncated):
        action = stationary_actions(env) if policy == "stationary" else equal_spacing_actions(env)
        next_observation, reward, terminated, truncated, info = env.step(action)
        returns += float(reward)
        length += 1
        rates.append(float(info["rate_e2e_bps"]))
        reward_terms = info["reward_terms"]
        for name in term_values:
            term_values[name].append(float(reward_terms[name]))
        interventions += int(np.any(np.asarray(info["intervention_norms"]) > 1e-9))
        positions = np.asarray(info["positions_m"], dtype=float)
        high_displacements.append(float(np.linalg.norm(positions[0] - initial_high)))
        low_displacements.append(float(np.linalg.norm(positions[-1] - initial_low)))
        relays = positions[1:-1]
        relay_path += float(np.mean(np.linalg.norm(relays - previous_relays, axis=1)))
        previous_relays = relays.copy()
        min_capacities.append(float(np.min(info["hop_capacities_bps"])))
        max_distances.append(float(np.max(info["hop_distances_m"])))
        if terminated:
            failure_reason = str(info.get("failure_reason", "unknown failure"))
        _ = next_observation
    values = [returns, *rates, *[value for values in term_values.values() for value in values], *high_displacements, *low_displacements, relay_path, *min_capacities, *max_distances]
    if not all(np.isfinite(value) for value in values):
        raise ValueError("scenario diagnostic values must be finite")
    return ScenarioEpisodeDiagnostic(
        radius, max_steps, policy, episode_index, seed, float(returns), length, bool(terminated), bool(truncated), failure_reason,
        float(np.mean(rates)), float(np.min(rates)), *(float(np.mean(term_values[name])) for name in term_values),
        float(interventions / length), float(np.mean(high_displacements)), float(np.max(high_displacements)),
        float(np.mean(low_displacements)), float(np.max(low_displacements)), relay_path,
        float(np.mean(min_capacities)), float(np.mean(max_distances)),
    )


def _summary(radius: float, max_steps: int, policy: str, episodes: list[ScenarioEpisodeDiagnostic]) -> ScenarioDiagnosticSummary:
    def mean(name: str) -> float:
        return float(np.mean([getattr(item, name) for item in episodes]))
    returns = np.asarray([item.episode_return for item in episodes], dtype=float)
    values = [np.mean(returns), np.std(returns), *[mean(name) for name in ("mean_rate_e2e_bps", "min_rate_e2e_bps", "mean_rate_reward", "mean_link_cost", "mean_separation_cost", "mean_intervention_cost", "mean_motion_cost", "intervention_rate", "mean_high_displacement_m", "mean_low_displacement_m", "relay_path_length_m", "mean_min_hop_capacity_bps", "mean_max_hop_distance_m", "episode_length")]]
    values.append(float(np.mean([item.terminated for item in episodes])))
    if not all(np.isfinite(value) for value in values):
        raise ValueError("scenario diagnostic summary must be finite")
    termination_rate = values[-1]
    return ScenarioDiagnosticSummary(radius, max_steps, policy, len(episodes), termination_rate, values[0], values[1], *values[2:-1])


def diagnose_scenarios(base_config: EnvironmentConfig, config: ScenarioDiagnosticConfig) -> ScenarioDiagnosticResult:
    if not isinstance(base_config, EnvironmentConfig) or not isinstance(config, ScenarioDiagnosticConfig):
        raise ValueError("base_config and config have incompatible types")
    episode_results: list[ScenarioEpisodeDiagnostic] = []
    summaries: list[ScenarioDiagnosticSummary] = []
    for radius in config.waypoint_radii_m:
        for max_steps in config.max_steps_values:
            scenario = _scenario_config(base_config, radius, max_steps)
            for policy in config.policies:
                episodes = [_episode(MultiRelayEnvironment(copy.deepcopy(scenario)), policy, int(config.seed + episode_index), radius, max_steps, episode_index) for episode_index in range(config.episodes)]
                episode_results.extend(episodes)
                summaries.append(_summary(radius, max_steps, policy, episodes))
    return ScenarioDiagnosticResult(config, tuple(episode_results), tuple(summaries))


__all__ = ["ScenarioDiagnosticConfig", "ScenarioDiagnosticResult", "ScenarioEpisodeDiagnostic", "ScenarioDiagnosticSummary", "diagnose_scenarios"]
