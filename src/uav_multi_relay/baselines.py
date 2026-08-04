"""Deterministic non-learning baselines for relay control."""

from __future__ import annotations

import copy

import numpy as np

from .environment import MultiRelayEnvironment


def equal_spacing_actions(env: MultiRelayEnvironment) -> np.ndarray:
    """Return normalized actions that steer relays towards equal H-to-L spacing."""
    if not isinstance(env, MultiRelayEnvironment):
        raise ValueError("env must be a MultiRelayEnvironment")
    high = env.high_state.position_m
    low = env.low_state.position_m
    actions = np.zeros((env.config.num_relays, 3), dtype=float)
    for index, relay in enumerate(env.relay_states, start=1):
        target = high + index / (env.config.num_relays + 1) * (low - high)
        actions[index - 1] = _direction_to_action(target - relay.position_m)
    return np.clip(actions, -1.0, 1.0)


def stationary_actions(env: MultiRelayEnvironment) -> np.ndarray:
    """Return a zero action for every relay."""
    if not isinstance(env, MultiRelayEnvironment):
        raise ValueError("env must be a MultiRelayEnvironment")
    return np.zeros((env.config.num_relays, 3), dtype=float)


def weighted_spacing_actions(
    env: MultiRelayEnvironment, hop_weights: object | None = None
) -> np.ndarray:
    """Steer relays to positions divided by cumulative positive hop weights."""
    if not isinstance(env, MultiRelayEnvironment):
        raise ValueError("env must be a MultiRelayEnvironment")
    hop_count = env.config.num_relays + 1
    if hop_weights is None:
        weights = np.ones(hop_count, dtype=float)
    else:
        weights = np.asarray(hop_weights, dtype=float)
        if weights.shape != (hop_count,) or not np.all(np.isfinite(weights)):
            raise ValueError(f"hop_weights must be finite with shape ({hop_count},)")
        if np.any(weights <= 0.0):
            raise ValueError("hop_weights must be strictly positive")
    cumulative = np.cumsum(weights)
    total = float(cumulative[-1])
    high = env.high_state.position_m
    low = env.low_state.position_m
    actions = np.zeros((env.config.num_relays, 3), dtype=float)
    for index, relay in enumerate(env.relay_states, start=1):
        fraction = cumulative[index - 1] / total
        target = high + fraction * (low - high)
        actions[index - 1] = _direction_to_action(target - relay.position_m)
    return np.clip(actions, -1.0, 1.0)


def greedy_one_step_actions(
    env: MultiRelayEnvironment, sweeps: int = 2
) -> np.ndarray:
    """Choose actions by deterministic coordinate search over one-step copies."""
    if not isinstance(env, MultiRelayEnvironment):
        raise ValueError("env must be a MultiRelayEnvironment")
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 0:
        raise ValueError("sweeps must be a nonnegative integer")

    best = stationary_actions(env)
    best_score = _evaluate_rate(env, best)
    equal = equal_spacing_actions(env)
    equal_score = _evaluate_rate(env, equal)
    if equal_score > best_score:
        best = equal.copy()
        best_score = equal_score

    values = (-1.0, -0.5, 0.0, 0.5, 1.0)
    for _ in range(sweeps):
        for relay_index in range(env.config.num_relays):
            for dimension in range(3):
                for value in values:
                    candidate = best.copy()
                    candidate[relay_index, dimension] = value
                    score = _evaluate_rate(env, candidate)
                    if score > best_score:
                        best = candidate
                        best_score = score
    return np.clip(best, -1.0, 1.0)


def _direction_to_action(displacement: np.ndarray) -> np.ndarray:
    distance = float(np.linalg.norm(displacement))
    return np.zeros(3, dtype=float) if distance == 0.0 else displacement / distance


def _evaluate_rate(env: MultiRelayEnvironment, actions: np.ndarray) -> float:
    """Evaluate only through the public environment step on an isolated copy."""
    candidate_env = copy.deepcopy(env)
    try:
        _, _, terminated, _, info = candidate_env.step(actions)
    except (RuntimeError, ValueError):
        return float("-inf")
    if terminated:
        return float("-inf")
    score = float(info["rate_e2e_bps"])
    return score if np.isfinite(score) else float("-inf")
