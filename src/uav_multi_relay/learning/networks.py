"""Shared stochastic actor and centralized twin-Q network definitions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(current, hidden_dim), nn.ReLU()))
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def _validate_hidden_dims(hidden_dims: Sequence[int]) -> tuple[int, ...]:
    values = tuple(hidden_dims)
    if not values or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("hidden_dims must contain positive integers")
    return values


def _finite_tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


class SharedGaussianActor(nn.Module):
    """One Gaussian policy shared across all relay observations."""

    def __init__(
        self,
        local_observation_dim: int = 23,
        action_dim: int = 3,
        hidden_dims: Sequence[int] = (256, 256),
        log_std_bounds: tuple[float, float] = (-20.0, 2.0),
    ) -> None:
        super().__init__()
        if local_observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation and action dimensions must be positive")
        hidden = _validate_hidden_dims(hidden_dims)
        if len(log_std_bounds) != 2 or not all(torch.isfinite(torch.tensor(value)) for value in log_std_bounds):
            raise ValueError("log_std_bounds must contain two finite values")
        if log_std_bounds[0] >= log_std_bounds[1]:
            raise ValueError("log_std_bounds must be increasing")
        self.local_observation_dim = int(local_observation_dim)
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_bounds[0])
        self.log_std_max = float(log_std_bounds[1])
        backbone_layers: list[nn.Module] = []
        current_dim = self.local_observation_dim
        for hidden_dim in hidden:
            backbone_layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*backbone_layers)
        self.mean_head = nn.Linear(hidden[-1], self.action_dim)
        self.log_std_head = nn.Linear(hidden[-1], self.action_dim)

    def forward(self, local_observations: Tensor) -> tuple[Tensor, Tensor]:
        observations = _finite_tensor(local_observations, "local_observations")
        if observations.ndim != 3 or observations.shape[-1] != self.local_observation_dim:
            raise ValueError(
                "local_observations must have shape (batch, num_relays, local_observation_dim)"
            )
        observations = observations.to(dtype=self.mean_head.weight.dtype, device=self.mean_head.weight.device)
        features = self.backbone(observations)
        mean = self.mean_head(features)
        raw_log_std = self.log_std_head(features)
        if not torch.isfinite(mean).all():
            raise ValueError("actor mean is non-finite")
        if not torch.isfinite(raw_log_std).all():
            raise ValueError("actor log_std is non-finite")
        log_std = raw_log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self, local_observations: Tensor, deterministic: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Sample tanh-squashed actions and their per-relay log probabilities."""
        mean, log_std = self.forward(local_observations)
        standard_deviation = log_std.exp()
        distribution = Normal(mean, standard_deviation)
        pre_tanh = mean if deterministic else distribution.rsample()
        actions = torch.tanh(pre_tanh)
        correction = torch.log(1.0 - actions.square() + 1e-6)
        log_probability = (distribution.log_prob(pre_tanh) - correction).sum(
            dim=-1, keepdim=True
        )
        return actions, log_probability

    def evaluate_actions(self, local_observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate requested tanh-squashed actions under this policy.

        Returns joint and per-relay log probabilities plus a finite entropy estimate.
        Clamping is only used for the inverse tanh at the closed action bounds.
        """
        mean, log_std = self.forward(local_observations)
        checked_actions = _finite_tensor(actions, "actions")
        if checked_actions.shape != mean.shape or torch.any(checked_actions < -1.0) or torch.any(checked_actions > 1.0):
            raise ValueError("actions must have actor output shape and lie within [-1, 1]")
        checked_actions = checked_actions.to(dtype=mean.dtype, device=mean.device)
        bounded = checked_actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(bounded)
        distribution = Normal(mean, log_std.exp())
        correction = torch.log(1.0 - bounded.square() + 1e-6)
        per_relay = (distribution.log_prob(pre_tanh) - correction).sum(dim=-1, keepdim=True)
        joint = per_relay.sum(dim=1)
        entropy_estimate = -joint
        if not torch.isfinite(joint).all() or not torch.isfinite(per_relay).all() or not torch.isfinite(entropy_estimate).all():
            raise ValueError("action evaluation is non-finite")
        return joint, per_relay, entropy_estimate


class CentralizedValueCritic(nn.Module):
    """One value estimate from the centralized global state."""

    def __init__(self, global_state_dim: int, hidden_dims: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        if global_state_dim <= 0:
            raise ValueError("global_state_dim must be positive")
        self.global_state_dim = int(global_state_dim)
        self.hidden_dims = _validate_hidden_dims(hidden_dims)
        self.value_net = _mlp(self.global_state_dim, self.hidden_dims, 1)

    def forward(self, global_state: Tensor) -> Tensor:
        state = _finite_tensor(global_state, "global_state")
        if state.ndim != 2 or state.shape[-1] != self.global_state_dim:
            raise ValueError("global_state must have shape (batch, global_state_dim)")
        state = state.to(dtype=self.value_net[0].weight.dtype, device=self.value_net[0].weight.device)
        value = self.value_net(state)
        if not torch.isfinite(value).all():
            raise ValueError("value critic output is non-finite")
        return value


class CentralizedTwinCritic(nn.Module):
    """Two independent Q networks over global state and flattened joint actions."""

    def __init__(
        self,
        global_state_dim: int,
        num_relays: int,
        action_dim: int = 3,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if global_state_dim <= 0 or num_relays <= 0 or action_dim <= 0:
            raise ValueError("state, relay, and action dimensions must be positive")
        hidden = _validate_hidden_dims(hidden_dims)
        self.global_state_dim = int(global_state_dim)
        self.num_relays = int(num_relays)
        self.action_dim = int(action_dim)
        input_dim = self.global_state_dim + self.num_relays * self.action_dim
        self.q1_net = _mlp(input_dim, hidden, 1)
        self.q2_net = _mlp(input_dim, hidden, 1)

    def forward(self, global_state: Tensor, joint_actions: Tensor) -> tuple[Tensor, Tensor]:
        state = _finite_tensor(global_state, "global_state")
        actions = _finite_tensor(joint_actions, "joint_actions")
        if state.ndim != 2 or state.shape[-1] != self.global_state_dim:
            raise ValueError("global_state must have shape (batch, global_state_dim)")
        if (
            actions.ndim != 3
            or actions.shape[0] != state.shape[0]
            or actions.shape[1] != self.num_relays
            or actions.shape[2] != self.action_dim
        ):
            raise ValueError("joint_actions must have shape (batch, num_relays, action_dim)")
        state = state.to(dtype=self.q1_net[0].weight.dtype, device=self.q1_net[0].weight.device)
        actions = actions.to(dtype=state.dtype, device=state.device)
        inputs = torch.cat((state, actions.flatten(start_dim=1)), dim=-1)
        q1 = self.q1_net(inputs)
        q2 = self.q2_net(inputs)
        if not torch.isfinite(q1).all() or not torch.isfinite(q2).all():
            raise ValueError("critic output is non-finite")
        return q1, q2
