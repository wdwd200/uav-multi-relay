"""Shared replay collection and update loop for deterministic MARL agents."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import numpy as np

from ..environment import MultiRelayEnvironment
from ..learning import (
    DeterministicUpdateMetrics,
    MultiAgentReplayBuffer,
    ParameterSharingMADDPG,
    ParameterSharingMATD3,
)
from .trainer import _observation_arrays

DeterministicAgent = ParameterSharingMADDPG | ParameterSharingMATD3


@dataclass(frozen=True)
class DeterministicTrainingConfig:
    total_environment_steps: int
    replay_capacity: int = 100_000
    batch_size: int = 256
    random_action_steps: int = 2_000
    update_after_steps: int = 2_000
    updates_per_step: int = 1
    exploration_noise_std: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("total_environment_steps", "replay_capacity", "batch_size", "updates_per_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("random_action_steps", "update_after_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not np.isfinite(self.exploration_noise_std) or self.exploration_noise_std < 0:
            raise ValueError("exploration_noise_std must be finite and non-negative")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size must not exceed replay_capacity")


@dataclass(frozen=True)
class DeterministicTrainingProgress:
    environment_steps: int
    total_updates: int
    completed_episodes: int
    replay_size: int
    mean_rate_e2e_bps: float
    termination_event_rate_per_step: float
    terminated_episode_rate: float
    mean_episode_length: float
    mean_episode_return: float
    intervention_rate: float
    requested_applied_mismatch_rate: float
    last_update_metrics: DeterministicUpdateMetrics | None


@dataclass(frozen=True)
class DeterministicTrainingSummary:
    total_environment_steps: int
    total_updates: int
    completed_episodes: int
    episode_returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    mean_rate_e2e_bps: float
    termination_event_rate_per_step: float
    terminated_episode_rate: float
    mean_episode_length: float
    mean_episode_return: float
    intervention_rate: float
    requested_applied_mismatch_rate: float
    last_update_metrics: DeterministicUpdateMetrics | None


def _validate_inputs(env: MultiRelayEnvironment, agent: DeterministicAgent,
                     replay_buffer: MultiAgentReplayBuffer,
                     config: DeterministicTrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(agent, (ParameterSharingMADDPG, ParameterSharingMATD3)):
        raise ValueError("agent must be a deterministic MARL agent")
    if replay_buffer.capacity != config.replay_capacity:
        raise ValueError("replay_buffer.capacity must equal config.replay_capacity")
    observation, _ = env.reset(seed=config.seed)
    local, global_state = _observation_arrays(observation)
    if local.shape != (agent.num_relays, agent.local_observation_dim) or global_state.shape != (agent.global_state_dim,):
        raise ValueError("agent and environment observations are incompatible")
    return local, global_state


def _progress(steps: int, updates: int, completed: int, terminations: int,
              rates: list[float], returns: list[float], lengths: list[int],
              interventions: int, mismatches: int, replay_size: int,
              metrics: DeterministicUpdateMetrics | None) -> DeterministicTrainingProgress:
    terminated_episodes = terminations / completed if completed else 0.0
    return DeterministicTrainingProgress(
        steps, updates, completed, replay_size, float(np.mean(rates)),
        terminations / steps, terminated_episodes,
        float(np.mean(lengths)) if lengths else 0.0,
        float(np.mean(returns)) if returns else 0.0,
        interventions / steps, mismatches / steps, metrics,
    )


def train_deterministic(env: MultiRelayEnvironment, agent: DeterministicAgent,
                        replay_buffer: MultiAgentReplayBuffer,
                        config: DeterministicTrainingConfig, *,
                        progress_interval_steps: int | None = None,
                        progress_callback: Callable[[DeterministicTrainingProgress], None] | None = None) -> DeterministicTrainingSummary:
    """Collect exactly configured steps, retaining safety-filtered actions in replay."""
    if not isinstance(config, DeterministicTrainingConfig):
        raise ValueError("config must be a DeterministicTrainingConfig")
    if progress_callback is not None and (
        not callable(progress_callback) or not isinstance(progress_interval_steps, Integral) or progress_interval_steps <= 0
    ):
        raise ValueError("progress callback requires a positive interval")
    local, global_state = _validate_inputs(env, agent, replay_buffer, config)
    rng = np.random.default_rng(config.seed)
    returns: list[float] = []
    lengths: list[int] = []
    rates: list[float] = []
    current_return = 0.0
    current_length = completed = updates = terminations = interventions = mismatches = 0
    last_metrics: DeterministicUpdateMetrics | None = None

    for step in range(config.total_environment_steps):
        if step < config.random_action_steps:
            requested = rng.uniform(-1.0, 1.0, (agent.num_relays, agent.action_dim)).astype(np.float32)
        else:
            noise = rng.normal(0.0, config.exploration_noise_std, (agent.num_relays, agent.action_dim))
            requested = np.clip(agent.act(local) + noise, -1.0, 1.0).astype(np.float32)
        next_observation, reward, terminated, truncated, info = env.step(requested)
        next_local, next_global = _observation_arrays(next_observation)
        applied = np.asarray(info["applied_relay_actions"], dtype=np.float32)
        replay_buffer.add(local, global_state, applied, reward, next_local, next_global, terminated, truncated)

        rate = float(info["rate_e2e_bps"])
        intervention_norms = np.asarray(info["intervention_norms"], dtype=float)
        mismatch_norms = np.linalg.norm(requested - applied, axis=1)
        if not all((np.isfinite(reward), np.isfinite(rate), np.all(np.isfinite(intervention_norms)), np.all(np.isfinite(mismatch_norms)))):
            raise ValueError("environment returned non-finite training values")
        rates.append(rate)
        current_return += float(reward)
        current_length += 1
        terminations += int(terminated)
        interventions += int(np.any(intervention_norms > 1e-9))
        mismatches += int(np.any(mismatch_norms > 1e-6))

        if step + 1 >= config.update_after_steps and replay_buffer.size >= config.batch_size:
            for _ in range(config.updates_per_step):
                last_metrics = agent.update(replay_buffer.sample(config.batch_size, device=agent.device))
                updates += 1
        local, global_state = next_local, next_global
        if terminated or truncated:
            returns.append(current_return)
            lengths.append(current_length)
            completed += 1
            current_return = 0.0
            current_length = 0
            observation, _ = env.reset(seed=config.seed + completed)
            local, global_state = _observation_arrays(observation)

        environment_steps = step + 1
        if progress_callback is not None and (
            environment_steps % int(progress_interval_steps) == 0 or environment_steps == config.total_environment_steps
        ):
            progress_callback(_progress(environment_steps, updates, completed, terminations, rates, returns, lengths, interventions, mismatches, replay_buffer.size, last_metrics))

    progress = _progress(config.total_environment_steps, updates, completed, terminations, rates, returns, lengths, interventions, mismatches, replay_buffer.size, last_metrics)
    values = (progress.mean_rate_e2e_bps, progress.termination_event_rate_per_step, progress.terminated_episode_rate, progress.mean_episode_length, progress.mean_episode_return, progress.intervention_rate, progress.requested_applied_mismatch_rate, *returns)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("training summary contains non-finite values")
    return DeterministicTrainingSummary(
        config.total_environment_steps, updates, completed, tuple(returns), tuple(lengths),
        progress.mean_rate_e2e_bps, progress.termination_event_rate_per_step,
        progress.terminated_episode_rate, progress.mean_episode_length,
        progress.mean_episode_return, progress.intervention_rate,
        progress.requested_applied_mismatch_rate, last_metrics,
    )


__all__ = ["DeterministicTrainingConfig", "DeterministicTrainingProgress", "DeterministicTrainingSummary", "train_deterministic"]
