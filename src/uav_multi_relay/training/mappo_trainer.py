"""On-policy rollout collection and MAPPO update scheduling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..environment import MultiRelayEnvironment
from ..learning import MAPPOAgent, MAPPOUpdateMetrics, MAPPORollout


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _observation_arrays(observation: object) -> tuple[np.ndarray, np.ndarray]:
    """Return finite local and global observations in the training dtypes."""
    if not isinstance(observation, dict) or "local" not in observation or "global" not in observation:
        raise ValueError("observation must contain local and global arrays")
    local = np.asarray(observation["local"], dtype=np.float32)
    global_state = np.asarray(observation["global"], dtype=np.float32)
    if local.ndim != 2 or global_state.ndim != 1:
        raise ValueError("observation arrays must have ranks 2 and 1")
    if not np.all(np.isfinite(local)) or not np.all(np.isfinite(global_state)):
        raise ValueError("observation arrays must be finite")
    return local, global_state


@dataclass(frozen=True)
class MAPPOTrainingConfig:
    total_environment_steps: int = 10_000
    rollout_steps: int = 1_024
    seed: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.total_environment_steps, "total_environment_steps")
        _positive_int(self.rollout_steps, "rollout_steps")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class MAPPOTrainingProgress:
    environment_steps: int
    total_updates: int
    completed_episodes: int
    mean_rate_e2e_bps: float
    termination_rate: float
    intervention_rate: float
    requested_applied_mismatch_rate: float
    last_update_metrics: MAPPOUpdateMetrics | None


@dataclass(frozen=True)
class MAPPOTrainingSummary:
    total_environment_steps: int
    total_updates: int
    completed_episodes: int
    episode_returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    mean_rate_e2e_bps: float
    termination_rate: float
    intervention_rate: float
    requested_applied_mismatch_rate: float
    discarded_partial_rollout_steps: int
    last_update_metrics: MAPPOUpdateMetrics | None


def train_mappo(
    env: MultiRelayEnvironment,
    agent: MAPPOAgent,
    config: MAPPOTrainingConfig,
    *,
    progress_interval_steps: int | None = None,
    progress_callback: Callable[[MAPPOTrainingProgress], None] | None = None,
) -> MAPPOTrainingSummary:
    """Collect requested-action rollouts and update only when a rollout is full."""
    if not isinstance(agent, MAPPOAgent) or not isinstance(config, MAPPOTrainingConfig):
        raise ValueError("agent and config have incompatible types")
    if progress_callback is not None and (not callable(progress_callback) or progress_interval_steps is None):
        raise ValueError("progress callback requires a positive interval")
    if progress_interval_steps is not None:
        _positive_int(progress_interval_steps, "progress_interval_steps")

    observation, _ = env.reset(seed=config.seed)
    local, global_state = _observation_arrays(observation)
    if local.shape != (agent.num_relays, agent.local_observation_dim) or global_state.shape != (agent.global_state_dim,):
        raise ValueError("agent and environment observations are incompatible")
    rollout = MAPPORollout(config.rollout_steps, agent.num_relays, agent.local_observation_dim, agent.global_state_dim, agent.action_dim)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    rates: list[float] = []
    completed_episodes = 0
    terminated_episodes = 0
    interventions = 0
    mismatch_events = 0
    current_return = 0.0
    current_length = 0
    total_updates = 0
    last_metrics: MAPPOUpdateMetrics | None = None

    for index in range(config.total_environment_steps):
        requested, old_per_relay_log_probability, value = agent.act_with_stats(local, global_state)
        next_observation, reward, terminated, truncated, info = env.step(requested)
        next_local, next_global = _observation_arrays(next_observation)
        next_value = agent.value(next_global)
        applied = np.asarray(info["applied_relay_actions"], dtype=np.float32)
        rollout.add(local, global_state, requested, applied, old_per_relay_log_probability, reward, value, next_value, terminated, truncated)

        rate = float(info["rate_e2e_bps"])
        intervention_norms = np.asarray(info["intervention_norms"], dtype=float)
        mismatch = np.linalg.norm(requested.astype(float) - applied.astype(float), axis=1)
        if not np.isfinite(rate) or not np.all(np.isfinite(intervention_norms)) or not np.all(np.isfinite(mismatch)):
            raise ValueError("environment rollout diagnostics are non-finite")
        rates.append(rate)
        interventions += int(np.any(intervention_norms > 1e-9))
        mismatch_events += int(np.any(mismatch > 1e-6))
        current_return += float(reward)
        current_length += 1

        if rollout.is_full:
            last_metrics = agent.update(rollout)
            total_updates += 1
            rollout.clear()

        local, global_state = next_local, next_global
        if terminated or truncated:
            episode_returns.append(float(current_return))
            episode_lengths.append(current_length)
            completed_episodes += 1
            terminated_episodes += int(terminated)
            current_return = 0.0
            current_length = 0
            observation, _ = env.reset(seed=config.seed + completed_episodes)
            local, global_state = _observation_arrays(observation)

        environment_steps = index + 1
        if progress_callback is not None and (environment_steps % int(progress_interval_steps) == 0 or environment_steps == config.total_environment_steps):
            progress_callback(MAPPOTrainingProgress(environment_steps, total_updates, completed_episodes, float(np.mean(rates)), float(terminated_episodes / completed_episodes) if completed_episodes else 0.0, float(interventions / environment_steps), float(mismatch_events / environment_steps), last_metrics))

    termination_rate = float(terminated_episodes / completed_episodes) if completed_episodes else 0.0
    values = [float(np.mean(rates)), termination_rate, float(interventions / config.total_environment_steps), float(mismatch_events / config.total_environment_steps), *episode_returns]
    if not all(np.isfinite(value) for value in values):
        raise ValueError("training summary contains non-finite values")
    return MAPPOTrainingSummary(config.total_environment_steps, total_updates, completed_episodes, tuple(episode_returns), tuple(episode_lengths), values[0], values[1], values[2], values[3], rollout.size, last_metrics)


__all__ = ["MAPPOTrainingConfig", "MAPPOTrainingProgress", "MAPPOTrainingSummary", "train_mappo"]
