"""Fixed-capacity multi-agent replay storage for executed transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ReplayBatch:
    """A sampled batch represented by detached PyTorch tensors."""

    local_observations: Tensor
    global_states: Tensor
    applied_actions: Tensor
    rewards: Tensor
    next_local_observations: Tensor
    next_global_states: Tensor
    terminated: Tensor
    truncated: Tensor


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _transition_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite with shape {shape}") from error
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array.copy()


def _flag(value: object, name: str) -> np.float32:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar in [0, 1]") from error
    if array.shape != () or not np.isfinite(array) or float(array) not in (0.0, 1.0):
        raise ValueError(f"{name} must be a finite scalar in [0, 1]")
    return np.float32(array)


class MultiAgentReplayBuffer:
    """Store normalized actions actually executed after safety filtering.

    Requested actions and physical velocities do not belong in ``applied_actions``.
    Termination and time-limit truncation remain separate transition fields.
    """

    def __init__(
        self,
        capacity: int,
        num_relays: int,
        local_observation_dim: int,
        global_state_dim: int,
        action_dim: int,
        seed: int | None = None,
    ) -> None:
        self.capacity = _positive_int(capacity, "capacity")
        self.num_relays = _positive_int(num_relays, "num_relays")
        self.local_observation_dim = _positive_int(
            local_observation_dim, "local_observation_dim"
        )
        self.global_state_dim = _positive_int(global_state_dim, "global_state_dim")
        self.action_dim = _positive_int(action_dim, "action_dim")
        self._position = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

        self.local_observations = np.zeros(
            (self.capacity, self.num_relays, self.local_observation_dim), dtype=np.float32
        )
        self.global_states = np.zeros((self.capacity, self.global_state_dim), dtype=np.float32)
        self.applied_actions = np.zeros(
            (self.capacity, self.num_relays, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_local_observations = np.zeros_like(self.local_observations)
        self.next_global_states = np.zeros_like(self.global_states)
        self.terminated = np.zeros((self.capacity, 1), dtype=np.float32)
        self.truncated = np.zeros((self.capacity, 1), dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    @property
    def position(self) -> int:
        return self._position

    def add(
        self,
        local_observations: object,
        global_state: object,
        applied_actions: object,
        reward: object,
        next_local_observations: object,
        next_global_state: object,
        terminated: object,
        truncated: object,
    ) -> None:
        """Copy one executed transition into the next circular slot."""
        local = _transition_array(
            local_observations,
            (self.num_relays, self.local_observation_dim),
            "local_observations",
        )
        global_state_array = _transition_array(
            global_state, (self.global_state_dim,), "global_state"
        )
        actions = _transition_array(
            applied_actions,
            (self.num_relays, self.action_dim),
            "applied_actions",
        )
        if np.any(actions < -1.0) or np.any(actions > 1.0):
            raise ValueError("applied_actions must be within [-1, 1]")
        try:
            reward_value = np.asarray(reward, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("reward must be a finite scalar") from error
        if reward_value.shape == ():
            reward_value = reward_value.reshape(1)
        reward_array = _transition_array(reward_value, (1,), "reward")
        next_local = _transition_array(
            next_local_observations,
            (self.num_relays, self.local_observation_dim),
            "next_local_observations",
        )
        next_global = _transition_array(
            next_global_state, (self.global_state_dim,), "next_global_state"
        )
        terminated_value = _flag(terminated, "terminated")
        truncated_value = _flag(truncated, "truncated")

        index = self._position
        self.local_observations[index] = local
        self.global_states[index] = global_state_array
        self.applied_actions[index] = actions
        self.rewards[index] = reward_array
        self.next_local_observations[index] = next_local
        self.next_global_states[index] = next_global
        self.terminated[index, 0] = terminated_value
        self.truncated[index, 0] = truncated_value
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: object | None = None) -> ReplayBatch:
        """Sample distinct valid transitions and return independent tensors."""
        batch_size = _positive_int(batch_size, "batch_size")
        if batch_size > self._size:
            raise ValueError("not enough valid transitions to sample batch_size")
        indices = self._rng.choice(self._size, size=batch_size, replace=False)

        def tensor(array: np.ndarray) -> Tensor:
            return torch.tensor(array[indices].copy(), dtype=torch.float32, device=device)

        return ReplayBatch(
            local_observations=tensor(self.local_observations),
            global_states=tensor(self.global_states),
            applied_actions=tensor(self.applied_actions),
            rewards=tensor(self.rewards),
            next_local_observations=tensor(self.next_local_observations),
            next_global_states=tensor(self.next_global_states),
            terminated=tensor(self.terminated),
            truncated=tensor(self.truncated),
        )

    store = add


__all__ = ["ReplayBatch", "MultiAgentReplayBuffer"]
