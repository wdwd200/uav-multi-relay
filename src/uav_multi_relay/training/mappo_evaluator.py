"""Deterministic MAPPO policy evaluation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..environment import MultiRelayEnvironment
from ..learning import MAPPOAgent
from .mappo_trainer import _observation_arrays


@dataclass(frozen=True)
class MAPPOEvaluationConfig:
    episodes: int = 10
    seed: int = 0
    def __post_init__(self) -> None:
        if isinstance(self.episodes, bool) or not isinstance(self.episodes, Integral) or self.episodes <= 0 or isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("episodes must be positive and seed must be an integer")


@dataclass(frozen=True)
class MAPPOEvaluationSummary:
    episodes: int
    mean_return: float
    return_std: float
    mean_rate_e2e_bps: float
    minimum_rate_e2e_bps: float
    terminated_episode_rate: float
    mean_episode_length: float
    mean_intervention_rate: float
    requested_applied_mismatch_rate: float


def evaluate_mappo(env: MultiRelayEnvironment, agent: MAPPOAgent, config: MAPPOEvaluationConfig) -> MAPPOEvaluationSummary:
    if not isinstance(agent, MAPPOAgent) or not isinstance(config, MAPPOEvaluationConfig):
        raise ValueError("agent and config have incompatible types")
    evaluation_env = copy.deepcopy(env); results: list[tuple[float, int, float, float, bool, float, float]] = []
    for episode in range(config.episodes):
        observation, _ = evaluation_env.reset(seed=config.seed + episode); local, global_state = _observation_arrays(observation)
        total_return = 0.0; rates: list[float] = []; length = interventions = mismatches = 0; terminated = truncated = False
        while not (terminated or truncated):
            requested = agent.act(local, deterministic=True)
            next_observation, reward, terminated, truncated, info = evaluation_env.step(requested)
            local, global_state = _observation_arrays(next_observation); applied = np.asarray(info["applied_relay_actions"], dtype=float)
            rate = float(info["rate_e2e_bps"]); norms = np.asarray(info["intervention_norms"], dtype=float); mismatch = np.linalg.norm(requested.astype(float) - applied, axis=1)
            if not np.isfinite(rate) or not np.all(np.isfinite(norms)) or not np.all(np.isfinite(mismatch)):
                raise ValueError("evaluation values are non-finite")
            total_return += float(reward); rates.append(rate); length += 1; interventions += int(np.any(norms > 1e-9)); mismatches += int(np.any(mismatch > 1e-6))
        results.append((total_return, length, float(np.mean(rates)), float(np.min(rates)), bool(terminated), interventions / length, mismatches / length))
    raw = np.asarray(results, dtype=float)
    values = [float(raw[:, 0].mean()), float(raw[:, 0].std()), float(raw[:, 2].mean()), float(raw[:, 3].min()), float(raw[:, 4].mean()), float(raw[:, 1].mean()), float(raw[:, 5].mean()), float(raw[:, 6].mean())]
    if not all(np.isfinite(value) for value in values):
        raise ValueError("evaluation summary is non-finite")
    return MAPPOEvaluationSummary(config.episodes, *values)


__all__ = ["MAPPOEvaluationConfig", "MAPPOEvaluationSummary", "evaluate_mappo"]
