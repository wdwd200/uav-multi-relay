"""Finite-horizon shooting and cross-entropy MPC baseline."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ..baselines import equal_spacing_actions, stationary_actions
from ..environment import MultiRelayEnvironment


def _finite_number(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite") from None
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 3
    population_size: int = 64
    iterations: int = 3
    elite_fraction: float = 0.2
    discount: float = 0.99
    initial_standard_deviation: float = 0.6
    minimum_standard_deviation: float = 0.05

    def __post_init__(self) -> None:
        for name in ("horizon", "population_size", "iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if value < (3 if name == "population_size" else 1):
                raise ValueError(f"{name} is out of range")
            object.__setattr__(self, name, int(value))
        elite = _finite_number(self.elite_fraction, "elite_fraction")
        discount = _finite_number(self.discount, "discount")
        initial = _finite_number(self.initial_standard_deviation, "initial_standard_deviation")
        minimum = _finite_number(self.minimum_standard_deviation, "minimum_standard_deviation")
        if not 0.0 < elite <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if not 0.0 < minimum <= initial:
            raise ValueError("standard deviations must satisfy 0 < minimum <= initial")
        object.__setattr__(self, "elite_fraction", elite)
        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "initial_standard_deviation", initial)
        object.__setattr__(self, "minimum_standard_deviation", minimum)


@dataclass(frozen=True)
class MPCSequenceEvaluation:
    discounted_return: float
    steps_evaluated: int
    terminated: bool
    truncated: bool
    mean_rate_e2e_bps: float


@dataclass(frozen=True)
class MPCPlan:
    first_action: np.ndarray
    action_sequence: np.ndarray
    predicted_return: float
    evaluation: MPCSequenceEvaluation


def _validate_discount(discount: object) -> float:
    value = _finite_number(discount, "discount")
    if not 0.0 <= value <= 1.0:
        raise ValueError("discount must be in [0, 1]")
    return value


def _validate_sequence(env: MultiRelayEnvironment, action_sequence: object) -> np.ndarray:
    try:
        array = np.asarray(action_sequence, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"action_sequence must be numeric with shape (horizon, {env.config.num_relays}, 3)"
        ) from error
    expected_rank = (env.config.num_relays, 3)
    if array.ndim != 3 or array.shape[1:] != expected_rank:
        raise ValueError(
            f"action_sequence must have shape (horizon, {env.config.num_relays}, 3)"
        )
    if array.shape[0] < 1 or not np.all(np.isfinite(array)):
        raise ValueError("action_sequence must be non-empty and finite")
    if np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError("action_sequence values must be within [-1, 1]")
    return array.copy()


def evaluate_action_sequence(
    env: MultiRelayEnvironment,
    action_sequence: object,
    discount: float = 0.99,
) -> MPCSequenceEvaluation:
    if not isinstance(env, MultiRelayEnvironment):
        raise ValueError("env must be a MultiRelayEnvironment")
    sequence = _validate_sequence(env, action_sequence)
    discount_value = _validate_discount(discount)
    prediction = copy.deepcopy(env)
    discounted_return = 0.0
    rates: list[float] = []
    terminated = truncated = False
    steps = 0
    for step_index, action in enumerate(sequence):
        _, reward, terminated, truncated, info = prediction.step(action)
        discounted_return += discount_value**step_index * float(reward)
        rate = float(info["rate_e2e_bps"])
        if not np.isfinite(rate) or not np.isfinite(reward):
            raise ValueError("environment returned a non-finite reward or rate")
        rates.append(rate)
        steps += 1
        if terminated or truncated:
            break
    return MPCSequenceEvaluation(
        discounted_return=float(discounted_return),
        steps_evaluated=steps,
        terminated=bool(terminated),
        truncated=bool(truncated),
        mean_rate_e2e_bps=float(np.mean(rates)) if rates else 0.0,
    )


def plan_mpc(
    env: MultiRelayEnvironment,
    config: MPCConfig | None = None,
    seed: int | None = 0,
) -> MPCPlan:
    if not isinstance(env, MultiRelayEnvironment):
        raise ValueError("env must be a MultiRelayEnvironment")
    config = MPCConfig() if config is None else config
    if not isinstance(config, MPCConfig):
        raise ValueError("config must be an MPCConfig")
    rng = np.random.default_rng(seed)
    equal_action = np.asarray(equal_spacing_actions(env), dtype=float)
    zero_action = np.asarray(stationary_actions(env), dtype=float)
    mean_sequence = np.repeat(equal_action[np.newaxis, :, :], config.horizon, axis=0)
    standard_deviation = np.full_like(mean_sequence, config.initial_standard_deviation)
    elite_count = max(1, int(np.ceil(config.population_size * config.elite_fraction)))
    best_sequence = None
    best_score = -np.inf
    best_evaluation = None
    for _ in range(config.iterations):
        candidates = rng.normal(mean_sequence, standard_deviation, size=(config.population_size, *mean_sequence.shape))
        candidates = np.clip(candidates, -1.0, 1.0)
        candidates[0] = mean_sequence
        candidates[1] = np.repeat(zero_action[np.newaxis, :, :], config.horizon, axis=0)
        candidates[2] = np.repeat(equal_action[np.newaxis, :, :], config.horizon, axis=0)
        evaluations = [evaluate_action_sequence(env, candidate, config.discount) for candidate in candidates]
        scores = np.asarray([evaluation.discounted_return for evaluation in evaluations])
        iteration_best = int(np.argmax(scores))
        if best_sequence is None or scores[iteration_best] > best_score:
            best_sequence = candidates[iteration_best].copy()
            best_score = float(scores[iteration_best])
            best_evaluation = evaluations[iteration_best]
        order = np.argsort(-scores, kind="stable")
        elites = candidates[order[:elite_count]]
        mean_sequence = np.clip(elites.mean(axis=0), -1.0, 1.0)
        standard_deviation = np.maximum(elites.std(axis=0), config.minimum_standard_deviation)
    if best_sequence is None or best_evaluation is None:
        raise RuntimeError("MPC did not evaluate any candidate sequence")
    best_sequence = np.asarray(best_sequence, dtype=float).copy()
    return MPCPlan(
        first_action=best_sequence[0].copy(),
        action_sequence=best_sequence.copy(),
        predicted_return=float(best_evaluation.discounted_return),
        evaluation=best_evaluation,
    )


def mpc_actions(
    env: MultiRelayEnvironment,
    config: MPCConfig | None = None,
    seed: int | None = 0,
) -> np.ndarray:
    return plan_mpc(env, config=config, seed=seed).first_action.copy()
