"""Finite environment-collection loop for parameter-sharing MASAC."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..environment import MultiRelayEnvironment
from ..learning import MASACUpdateMetrics, MultiAgentReplayBuffer, ParameterSharingMASAC


def _int(value: object, name: str, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if (positive and result <= 0) or (not positive and result < 0):
        bound = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {bound} integer")
    return result


@dataclass(frozen=True)
class MASACTrainingConfig:
    total_environment_steps: int = 10_000
    replay_capacity: int = 100_000
    batch_size: int = 256
    random_action_steps: int = 1_000
    update_after_steps: int = 1_000
    updates_per_step: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("total_environment_steps", "replay_capacity", "batch_size", "updates_per_step"):
            _int(getattr(self, name), name, positive=True)
        for name in ("random_action_steps", "update_after_steps"):
            _int(getattr(self, name), name, positive=False)
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("seed must be an integer")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size must be no greater than replay_capacity")


@dataclass(frozen=True)
class MASACTrainingSummary:
    total_environment_steps: int
    total_updates: int
    completed_episodes: int
    episode_returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    mean_rate_e2e_bps: float
    intervention_rate: float
    last_update_metrics: MASACUpdateMetrics | None


def _observation_arrays(observation: object) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(observation, dict) or "local" not in observation or "global" not in observation:
        raise ValueError("observation must contain local and global arrays")
    local = np.asarray(observation["local"], dtype=np.float32)
    global_state = np.asarray(observation["global"], dtype=np.float32)
    if local.ndim != 2 or global_state.ndim != 1 or not np.all(np.isfinite(local)) or not np.all(np.isfinite(global_state)):
        raise ValueError("observation arrays must be finite with rank 2 and rank 1")
    return local, global_state


def train_masac(
    env: MultiRelayEnvironment,
    agent: ParameterSharingMASAC,
    replay_buffer: MultiAgentReplayBuffer,
    config: MASACTrainingConfig,
) -> MASACTrainingSummary:
    """Collect exactly the configured number of steps and perform updates."""
    if not isinstance(config, MASACTrainingConfig):
        raise ValueError("config must be a MASACTrainingConfig")
    if replay_buffer.capacity != config.replay_capacity:
        raise ValueError("replay_buffer.capacity must equal config.replay_capacity")
    observation, _ = env.reset(seed=config.seed)
    local, global_state = _observation_arrays(observation)
    num_relays, local_dim = local.shape
    global_dim = global_state.shape[0]
    if num_relays < 1 or getattr(env.config, "num_relays", num_relays) != num_relays:
        raise ValueError("observation relay count does not match environment")
    expected = {
        "agent.num_relays": getattr(agent, "num_relays", None),
        "agent.local_observation_dim": getattr(agent, "local_observation_dim", None),
        "agent.global_state_dim": getattr(agent, "global_state_dim", None),
        "agent.action_dim": getattr(agent, "action_dim", None),
        "replay_buffer.num_relays": getattr(replay_buffer, "num_relays", None),
        "replay_buffer.local_observation_dim": getattr(replay_buffer, "local_observation_dim", None),
        "replay_buffer.global_state_dim": getattr(replay_buffer, "global_state_dim", None),
        "replay_buffer.action_dim": getattr(replay_buffer, "action_dim", None),
    }
    for name, value in expected.items():
        if value is None:
            raise ValueError(f"{name} is unavailable")
    if expected["agent.num_relays"] != num_relays or expected["replay_buffer.num_relays"] != num_relays:
        raise ValueError("relay count is inconsistent")
    if expected["agent.local_observation_dim"] != local_dim or expected["replay_buffer.local_observation_dim"] != local_dim:
        raise ValueError("local observation dimension is inconsistent")
    if expected["agent.global_state_dim"] != global_dim or expected["replay_buffer.global_state_dim"] != global_dim:
        raise ValueError("global state dimension is inconsistent")
    if expected["agent.action_dim"] != 3 or expected["replay_buffer.action_dim"] != 3:
        raise ValueError("action dimension must be 3")

    rng = np.random.default_rng(config.seed)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    current_return = 0.0
    current_length = 0
    completed_episodes = 0
    total_updates = 0
    rates: list[float] = []
    interventions = 0
    last_metrics: MASACUpdateMetrics | None = None

    for collected_steps in range(config.total_environment_steps):
        requested = (
            rng.uniform(-1.0, 1.0, size=(num_relays, 3)).astype(np.float32)
            if collected_steps < config.random_action_steps
            else agent.act(local, deterministic=False)
        )
        next_observation, reward, terminated, truncated, info = env.step(requested)
        next_local, next_global = _observation_arrays(next_observation)
        applied = np.asarray(info["applied_relay_actions"], dtype=np.float32)
        replay_buffer.add(local, global_state, applied, reward, next_local, next_global, terminated, truncated)
        current_return += float(reward)
        current_length += 1
        rate = float(info["rate_e2e_bps"])
        norms = np.asarray(info["intervention_norms"], dtype=float)
        if not np.isfinite(rate) or not np.all(np.isfinite(norms)):
            raise ValueError("environment statistics must be finite")
        rates.append(rate)
        interventions += int(np.any(norms > 1e-9))
        if collected_steps + 1 >= config.update_after_steps and replay_buffer.size >= config.batch_size:
            for _ in range(config.updates_per_step):
                last_metrics = agent.update(replay_buffer.sample(config.batch_size, device=getattr(agent, "device", None)))
                total_updates += 1
        local, global_state = next_local, next_global
        if terminated or truncated:
            episode_returns.append(float(current_return))
            episode_lengths.append(current_length)
            completed_episodes += 1
            current_return = 0.0
            current_length = 0
            observation, _ = env.reset(seed=config.seed + completed_episodes)
            local, global_state = _observation_arrays(observation)

    mean_rate = float(np.mean(rates)) if rates else 0.0
    intervention_rate = float(interventions / config.total_environment_steps)
    values = [mean_rate, intervention_rate, *episode_returns]
    if not all(np.isfinite(value) for value in values):
        raise ValueError("training summary contains non-finite values")
    return MASACTrainingSummary(
        total_environment_steps=config.total_environment_steps,
        total_updates=total_updates,
        completed_episodes=completed_episodes,
        episode_returns=tuple(episode_returns),
        episode_lengths=tuple(episode_lengths),
        mean_rate_e2e_bps=mean_rate,
        intervention_rate=intervention_rate,
        last_update_metrics=last_metrics,
    )


__all__ = ["MASACTrainingConfig", "MASACTrainingSummary", "train_masac"]
