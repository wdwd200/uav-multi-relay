"""Finite environment-collection loop for parameter-sharing MASAC."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from collections.abc import Callable

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


@dataclass(frozen=True)
class MASACTrainingProgress:
    environment_steps: int
    total_updates: int
    completed_episodes: int
    replay_size: int
    mean_rate_e2e_bps: float
    intervention_rate: float
    last_update_metrics: MASACUpdateMetrics | None
    interval_diagnostics: dict[str, object] | None = None


@dataclass
class _IntervalDiagnostics:
    """Per-log-interval collection statistics; all values remain JSON-safe floats."""

    requested: list[np.ndarray] = field(default_factory=list)
    applied: list[np.ndarray] = field(default_factory=list)
    velocity_mismatch: list[float] = field(default_factory=list)
    mismatch: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    intervention_events: int = 0
    mismatch_events: int = 0
    terminations: int = 0
    truncations: int = 0
    episode_lengths: list[int] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)
    failure_reasons: dict[str, int] = field(default_factory=dict)

    def add_step(self, requested: np.ndarray, info: dict[str, object], terminated: bool, truncated: bool) -> None:
        applied = np.asarray(info["applied_relay_actions"], dtype=float)
        requested_velocity = np.asarray(info["requested_relay_velocities_mps"], dtype=float)
        applied_velocity = np.asarray(info["applied_relay_velocities_mps"], dtype=float)
        scale = float(info["safety_scale"])
        mismatch = np.linalg.norm(np.asarray(requested, dtype=float) - applied, axis=1)
        velocity_mismatch = np.linalg.norm(requested_velocity - applied_velocity, axis=1)
        values = (requested, applied, requested_velocity, applied_velocity, mismatch, velocity_mismatch, np.asarray(info["intervention_norms"], dtype=float), np.asarray([scale]))
        if not all(np.all(np.isfinite(value)) for value in values):
            raise ValueError("training diagnostics contain non-finite action statistics")
        self.requested.append(np.asarray(requested, dtype=float).copy())
        self.applied.append(applied.copy())
        self.mismatch.extend(float(value) for value in mismatch)
        self.velocity_mismatch.extend(float(value) for value in velocity_mismatch)
        self.scales.append(scale)
        self.intervention_events += int(np.any(np.asarray(info["intervention_norms"], dtype=float) > 1e-9))
        self.mismatch_events += int(np.any(mismatch > 1e-6))
        self.terminations += int(terminated)
        self.truncations += int(truncated)
        if terminated:
            reason = str(info.get("failure_reason", "unknown"))
            self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1

    def add_episode(self, length: int, episode_return: float) -> None:
        self.episode_lengths.append(int(length))
        self.episode_returns.append(float(episode_return))

    def payload(self, start_step: int, end_step: int) -> dict[str, object]:
        if not self.requested:
            raise ValueError("cannot emit empty interval diagnostics")
        requested = np.concatenate([item.reshape(-1, item.shape[-1]) for item in self.requested])
        applied = np.concatenate([item.reshape(-1, item.shape[-1]) for item in self.applied])
        mismatch = np.asarray(self.mismatch, dtype=float)
        velocity_mismatch = np.asarray(self.velocity_mismatch, dtype=float)
        scales = np.asarray(self.scales, dtype=float)
        components = requested.shape[0]
        payload: dict[str, object] = {
            "interval_start_step": int(start_step), "interval_end_step": int(end_step),
            "requested_action_mean": float(requested.mean()), "requested_action_std": float(requested.std()), "requested_action_abs_mean": float(np.abs(requested).mean()),
            "applied_action_mean": float(applied.mean()), "applied_action_std": float(applied.std()), "applied_action_abs_mean": float(np.abs(applied).mean()),
            "requested_action_saturation_rate": float(np.mean(np.abs(requested) >= 0.95)), "applied_action_saturation_rate": float(np.mean(np.abs(applied) >= 0.95)),
            "action_mismatch_event_rate": float(self.mismatch_events / len(self.requested)),
            "action_mismatch_l2_mean": float(mismatch.mean()), "action_mismatch_l2_p95": float(np.quantile(mismatch, 0.95)), "action_mismatch_l2_max": float(mismatch.max()),
            "velocity_mismatch_l2_mean": float(velocity_mismatch.mean()), "velocity_mismatch_l2_p95": float(np.quantile(velocity_mismatch, 0.95)), "velocity_mismatch_l2_max": float(velocity_mismatch.max()),
            "safety_scale_mean": float(scales.mean()), "safety_scale_min": float(scales.min()), "safety_scale_lt_one_rate": float(np.mean(scales < 1.0)),
            "intervention_event_rate": float(self.intervention_events / len(self.requested)), "termination_count": int(self.terminations), "truncation_count": int(self.truncations),
            "episode_length_mean": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
            "episode_return_mean": float(np.mean(self.episode_returns)) if self.episode_returns else 0.0,
            "return_per_step": float(np.sum(self.episode_returns) / np.sum(self.episode_lengths)) if self.episode_lengths else 0.0,
            "failure_reason_counts": dict(self.failure_reasons), "action_components": int(components),
        }
        finite = [value for value in payload.values() if isinstance(value, float)]
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("interval diagnostics contain non-finite values")
        return payload


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
    *,
    progress_interval_steps: int | None = None,
    progress_callback: Callable[[MASACTrainingProgress], None] | None = None,
    diagnostic_interval_steps: int | None = None,
    failure_trace_callback: Callable[[dict[str, object]], None] | None = None,
) -> MASACTrainingSummary:
    """Collect exactly the configured number of steps and perform updates."""
    if not isinstance(config, MASACTrainingConfig):
        raise ValueError("config must be a MASACTrainingConfig")
    if replay_buffer.capacity != config.replay_capacity:
        raise ValueError("replay_buffer.capacity must equal config.replay_capacity")
    if progress_callback is not None:
        if not callable(progress_callback):
            raise ValueError("progress_callback must be callable")
        if (
            progress_interval_steps is None
            or isinstance(progress_interval_steps, bool)
            or not isinstance(progress_interval_steps, Integral)
            or progress_interval_steps <= 0
        ):
            raise ValueError("progress_interval_steps must be a positive integer when callback is provided")
    elif progress_interval_steps is not None and (
        isinstance(progress_interval_steps, bool)
        or not isinstance(progress_interval_steps, Integral)
        or progress_interval_steps <= 0
    ):
        raise ValueError("progress_interval_steps must be a positive integer")
    for name, value in (("diagnostic_interval_steps", diagnostic_interval_steps),):
        if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    if failure_trace_callback is not None and not callable(failure_trace_callback):
        raise ValueError("failure_trace_callback must be callable")
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
    interval = _IntervalDiagnostics() if diagnostic_interval_steps is not None else None
    interval_start_step = 1
    recent_trace: list[dict[str, object]] = []
    episode_seed = int(config.seed)

    def emit_progress(environment_steps: int) -> None:
        if progress_callback is None:
            return
        mean_rate = float(np.mean(rates)) if rates else 0.0
        intervention_rate = float(interventions / environment_steps) if environment_steps else 0.0
        progress_callback(MASACTrainingProgress(
            environment_steps=environment_steps,
            total_updates=total_updates,
            completed_episodes=completed_episodes,
            replay_size=replay_buffer.size,
            mean_rate_e2e_bps=mean_rate,
            intervention_rate=intervention_rate,
            last_update_metrics=last_metrics,
            interval_diagnostics=None,
        ))

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
        if interval is not None:
            interval.add_step(requested, info, terminated, truncated)
        if failure_trace_callback is not None:
            requested_action = np.asarray(info["requested_relay_actions"], dtype=float)
            applied_action = np.asarray(info["applied_relay_actions"], dtype=float)
            trace_step = {
                "requested_action": requested_action.tolist(), "applied_action": applied_action.tolist(),
                "action_mismatch_norm": np.linalg.norm(requested_action - applied_action, axis=1).astype(float).tolist(),
                "safety_scale": float(info["safety_scale"]), "positions": np.asarray(info["positions_m"], dtype=float).tolist(),
                "velocities": np.asarray(info["velocities_mps"], dtype=float).tolist(), "hop_distances": np.asarray(info["hop_distances_m"], dtype=float).tolist(),
                "rate": float(info["rate_e2e_bps"]), "reward_terms": {key: float(value) for key, value in dict(info["reward_terms"]).items()},
            }
            if not all(np.isfinite(value) for value in [trace_step["safety_scale"], trace_step["rate"], *trace_step["reward_terms"].values()]):
                raise ValueError("failure trace contains non-finite values")
            recent_trace.append(trace_step)
            recent_trace = recent_trace[-10:]
        if collected_steps + 1 >= config.update_after_steps and replay_buffer.size >= config.batch_size:
            for _ in range(config.updates_per_step):
                last_metrics = agent.update(replay_buffer.sample(config.batch_size, device=getattr(agent, "device", None)))
                total_updates += 1
        local, global_state = next_local, next_global
        if terminated or truncated:
            episode_returns.append(float(current_return))
            episode_lengths.append(current_length)
            completed_episodes += 1
            if interval is not None:
                interval.add_episode(current_length, current_return)
            if terminated and failure_trace_callback is not None:
                failure_trace_callback({
                    "episode_seed": episode_seed, "episode_index": completed_episodes - 1,
                    "termination_step": current_length, "failure_reason": str(info.get("failure_reason", "unknown")),
                    "last_10_requested_actions": [entry["requested_action"] for entry in recent_trace],
                    "last_10_applied_actions": [entry["applied_action"] for entry in recent_trace],
                    "last_10_action_mismatch_norms": [entry["action_mismatch_norm"] for entry in recent_trace],
                    "last_10_safety_scales": [entry["safety_scale"] for entry in recent_trace],
                    "last_10_positions": [entry["positions"] for entry in recent_trace],
                    "last_10_velocities": [entry["velocities"] for entry in recent_trace],
                    "last_10_hop_distances": [entry["hop_distances"] for entry in recent_trace],
                    "last_10_rates": [entry["rate"] for entry in recent_trace],
                    "last_10_reward_terms": [entry["reward_terms"] for entry in recent_trace],
                })
            current_return = 0.0
            current_length = 0
            observation, _ = env.reset(seed=config.seed + completed_episodes)
            episode_seed = int(config.seed + completed_episodes)
            local, global_state = _observation_arrays(observation)
            recent_trace = []
        environment_steps = collected_steps + 1
        if interval is not None and (environment_steps % diagnostic_interval_steps == 0 or environment_steps == config.total_environment_steps):
            diagnostic_payload = interval.payload(interval_start_step, environment_steps)
            if progress_callback is not None:
                mean_rate = float(np.mean(rates)) if rates else 0.0
                progress_callback(MASACTrainingProgress(environment_steps, total_updates, completed_episodes, replay_buffer.size, mean_rate, float(interventions / environment_steps), last_metrics, diagnostic_payload))
            interval = _IntervalDiagnostics()
            interval_start_step = environment_steps + 1
        if progress_callback is not None and (
            environment_steps % int(progress_interval_steps) == 0
            or environment_steps == config.total_environment_steps
        ):
            if diagnostic_interval_steps is None or environment_steps % diagnostic_interval_steps != 0:
                emit_progress(environment_steps)

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


__all__ = ["MASACTrainingConfig", "MASACTrainingProgress", "MASACTrainingSummary", "train_masac"]
