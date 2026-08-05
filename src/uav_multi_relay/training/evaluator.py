"""Deterministic, side-effect-free MASAC evaluation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..environment import MultiRelayEnvironment
from ..learning import ParameterSharingMASAC
from .trainer import _observation_arrays


@dataclass(frozen=True)
class MASACEvaluationConfig:
    episodes: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.episodes, bool) or not isinstance(self.episodes, Integral) or self.episodes <= 0:
            raise ValueError("episodes must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class MASACEpisodeResult:
    episode_return: float
    episode_length: int
    mean_rate_e2e_bps: float
    min_rate_e2e_bps: float
    intervention_rate: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class MASACEvaluationSummary:
    episodes: int
    mean_return: float
    return_std: float
    mean_rate_e2e_bps: float
    minimum_rate_e2e_bps: float
    mean_intervention_rate: float
    terminated_episode_rate: float
    episode_results: tuple[MASACEpisodeResult, ...]


def evaluate_masac(
    env: MultiRelayEnvironment,
    agent: ParameterSharingMASAC,
    config: MASACEvaluationConfig,
) -> MASACEvaluationSummary:
    if not isinstance(config, MASACEvaluationConfig):
        raise ValueError("config must be a MASACEvaluationConfig")
    evaluation_env = copy.deepcopy(env)
    results: list[MASACEpisodeResult] = []
    for episode_index in range(config.episodes):
        observation, _ = evaluation_env.reset(seed=config.seed + episode_index)
        local, global_state = _observation_arrays(observation)
        if local.shape != (agent.num_relays, agent.local_observation_dim) or global_state.shape != (agent.global_state_dim,):
            raise ValueError("checkpoint agent and environment observations are incompatible")
        total_return = 0.0
        rates: list[float] = []
        interventions = 0
        length = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(local, deterministic=True)
            next_observation, reward, terminated, truncated, info = evaluation_env.step(action)
            local, global_state = _observation_arrays(next_observation)
            rate = float(info["rate_e2e_bps"])
            norms = np.asarray(info["intervention_norms"], dtype=float)
            if not np.isfinite(rate) or not np.all(np.isfinite(norms)):
                raise ValueError("environment statistics must be finite")
            total_return += float(reward)
            rates.append(rate)
            interventions += int(np.any(norms > 1e-9))
            length += 1
        results.append(MASACEpisodeResult(
            episode_return=float(total_return),
            episode_length=length,
            mean_rate_e2e_bps=float(np.mean(rates)) if rates else 0.0,
            min_rate_e2e_bps=float(np.min(rates)) if rates else 0.0,
            intervention_rate=float(interventions / length) if length else 0.0,
            terminated=bool(terminated),
            truncated=bool(truncated),
        ))
    returns = np.asarray([result.episode_return for result in results], dtype=float)
    rates = np.asarray([result.mean_rate_e2e_bps for result in results], dtype=float)
    minimums = np.asarray([result.min_rate_e2e_bps for result in results], dtype=float)
    interventions = np.asarray([result.intervention_rate for result in results], dtype=float)
    values = [float(np.mean(returns)), float(np.std(returns)), float(np.mean(rates)), float(np.min(minimums)), float(np.mean(interventions)), float(np.mean([result.terminated for result in results]))]
    if not all(np.isfinite(value) for value in values):
        raise ValueError("evaluation summary contains non-finite values")
    return MASACEvaluationSummary(
        episodes=config.episodes,
        mean_return=values[0],
        return_std=values[1],
        mean_rate_e2e_bps=values[2],
        minimum_rate_e2e_bps=values[3],
        mean_intervention_rate=values[4],
        terminated_episode_rate=values[5],
        episode_results=tuple(results),
    )


__all__ = ["MASACEpisodeResult", "MASACEvaluationConfig", "MASACEvaluationSummary", "evaluate_masac"]
