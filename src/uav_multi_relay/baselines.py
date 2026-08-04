"""Deterministic non-learning baselines for relay control."""

from __future__ import annotations

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
        displacement = target - relay.position_m
        distance = float(np.linalg.norm(displacement))
        if distance > 0.0:
            actions[index - 1] = displacement / distance
    return np.clip(actions, -1.0, 1.0)
