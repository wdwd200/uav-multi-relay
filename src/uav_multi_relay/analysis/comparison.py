"""Fair, deterministic comparisons of MASAC and existing control baselines."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..baselines import equal_spacing_actions, greedy_one_step_actions, stationary_actions
from ..environment import MultiRelayEnvironment
from ..learning import ParameterSharingMASAC
from ..policies import MPCConfig, mpc_actions


_VALID_POLICIES = frozenset({"masac", "random", "stationary", "equal_spacing", "greedy", "mpc"})


@dataclass(frozen=True)
class PolicyComparisonConfig:
    episodes: int = 5
    seed: int = 20_000
    policies: tuple[str, ...] = ("masac", "random", "stationary", "equal_spacing", "greedy", "mpc")
    greedy_sweeps: int = 1
    mpc_config: MPCConfig = MPCConfig(horizon=2, population_size=8, iterations=2, elite_fraction=0.5)

    def __post_init__(self) -> None:
        if isinstance(self.episodes, bool) or not isinstance(self.episodes, Integral) or self.episodes <= 0:
            raise ValueError("episodes must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("seed must be an integer")
        if isinstance(self.greedy_sweeps, bool) or not isinstance(self.greedy_sweeps, Integral) or self.greedy_sweeps < 0:
            raise ValueError("greedy_sweeps must be a non-negative integer")
        if not isinstance(self.policies, tuple) or not self.policies:
            raise ValueError("policies must be a non-empty tuple")
        if any(name not in _VALID_POLICIES for name in self.policies) or len(set(self.policies)) != len(self.policies):
            raise ValueError("policies must contain distinct valid policy names")
        if not isinstance(self.mpc_config, MPCConfig):
            raise ValueError("mpc_config must be an MPCConfig")


@dataclass(frozen=True)
class PolicyEpisodeResult:
    policy: str
    episode_index: int
    episode_seed: int
    episode_return: float
    episode_length: int
    mean_rate_e2e_bps: float
    min_rate_e2e_bps: float
    intervention_rate: float
    terminated: bool
    truncated: bool
    mean_action_compute_time_s: float


@dataclass(frozen=True)
class PolicySummary:
    policy: str
    episodes: int
    mean_return: float
    return_std: float
    mean_rate_e2e_bps: float
    minimum_rate_e2e_bps: float
    mean_intervention_rate: float
    mean_episode_length: float
    terminated_episode_rate: float
    mean_action_compute_time_s: float


@dataclass(frozen=True)
class PolicyComparisonResult:
    config: PolicyComparisonConfig
    policy_summaries: tuple[PolicySummary, ...]
    episode_results: tuple[PolicyEpisodeResult, ...]


def _action(
    policy: str,
    env: MultiRelayEnvironment,
    agent: ParameterSharingMASAC,
    local_observation: np.ndarray,
    random_generator: np.random.Generator,
    config: PolicyComparisonConfig,
    episode_index: int,
    step_index: int,
) -> np.ndarray:
    if policy == "masac":
        return agent.act(local_observation, deterministic=True)
    if policy == "random":
        return random_generator.uniform(-1.0, 1.0, size=(env.config.num_relays, 3))
    if policy == "stationary":
        return stationary_actions(env)
    if policy == "equal_spacing":
        return equal_spacing_actions(env)
    if policy == "greedy":
        return greedy_one_step_actions(env, sweeps=int(config.greedy_sweeps))
    return mpc_actions(
        env,
        config=config.mpc_config,
        seed=int(config.seed + episode_index * env.config.max_steps + step_index),
    )


def _summary(policy: str, episodes: list[PolicyEpisodeResult]) -> PolicySummary:
    returns = np.asarray([item.episode_return for item in episodes], dtype=float)
    rates = np.asarray([item.mean_rate_e2e_bps for item in episodes], dtype=float)
    minimums = np.asarray([item.min_rate_e2e_bps for item in episodes], dtype=float)
    interventions = np.asarray([item.intervention_rate for item in episodes], dtype=float)
    lengths = np.asarray([item.episode_length for item in episodes], dtype=float)
    timings = np.asarray([item.mean_action_compute_time_s for item in episodes], dtype=float)
    values = (np.mean(returns), np.std(returns), np.mean(rates), np.min(minimums), np.mean(interventions), np.mean(lengths), np.mean([item.terminated for item in episodes]), np.mean(timings))
    if not all(np.isfinite(value) for value in values):
        raise ValueError("comparison statistics must be finite")
    return PolicySummary(policy, len(episodes), *(float(value) for value in values))


def compare_policies(
    env: MultiRelayEnvironment,
    agent: ParameterSharingMASAC,
    config: PolicyComparisonConfig,
) -> PolicyComparisonResult:
    """Evaluate each selected policy on an identical seeded trajectory set."""
    if not isinstance(env, MultiRelayEnvironment) or not isinstance(agent, ParameterSharingMASAC):
        raise ValueError("env and agent have incompatible types")
    if not isinstance(config, PolicyComparisonConfig):
        raise ValueError("config must be a PolicyComparisonConfig")
    if env.config.num_relays != agent.num_relays:
        raise ValueError("agent and environment relay counts are incompatible")
    all_results: list[PolicyEpisodeResult] = []
    summaries: list[PolicySummary] = []
    for policy in config.policies:
        policy_results: list[PolicyEpisodeResult] = []
        for episode_index in range(config.episodes):
            episode_seed = int(config.seed + episode_index)
            evaluation_env = copy.deepcopy(env)
            observation, _ = evaluation_env.reset(seed=episode_seed)
            local = np.asarray(observation["local"], dtype=np.float32)
            if local.shape != (agent.num_relays, agent.local_observation_dim):
                raise ValueError("agent and environment local observations are incompatible")
            random_generator = np.random.default_rng(episode_seed)
            total_return = 0.0
            rates: list[float] = []
            interventions = 0
            total_action_time = 0.0
            step_index = 0
            terminated = truncated = False
            while not (terminated or truncated):
                started = time.perf_counter()
                action = _action(policy, evaluation_env, agent, local, random_generator, config, episode_index, step_index)
                elapsed = time.perf_counter() - started
                if not np.isfinite(elapsed) or elapsed < 0.0:
                    raise ValueError("action computation time must be finite and non-negative")
                next_observation, reward, terminated, truncated, info = evaluation_env.step(action)
                rate = float(info["rate_e2e_bps"])
                norms = np.asarray(info["intervention_norms"], dtype=float)
                if not np.isfinite(reward) or not np.isfinite(rate) or not np.all(np.isfinite(norms)):
                    raise ValueError("environment returned non-finite comparison statistics")
                total_return += float(reward)
                rates.append(rate)
                interventions += int(np.any(norms > 1e-9))
                total_action_time += elapsed
                local = np.asarray(next_observation["local"], dtype=np.float32)
                step_index += 1
            result = PolicyEpisodeResult(
                policy=policy,
                episode_index=episode_index,
                episode_seed=episode_seed,
                episode_return=float(total_return),
                episode_length=step_index,
                mean_rate_e2e_bps=float(np.mean(rates)),
                min_rate_e2e_bps=float(np.min(rates)),
                intervention_rate=float(interventions / step_index),
                terminated=bool(terminated),
                truncated=bool(truncated),
                mean_action_compute_time_s=float(total_action_time / step_index),
            )
            policy_results.append(result)
            all_results.append(result)
        summaries.append(_summary(policy, policy_results))
    return PolicyComparisonResult(config, tuple(summaries), tuple(all_results))


__all__ = ["PolicyComparisonConfig", "PolicyComparisonResult", "PolicyEpisodeResult", "PolicySummary", "compare_policies"]
