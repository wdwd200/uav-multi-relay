"""Parameter-sharing MASAC update primitives without an environment loop."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Real

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .networks import CentralizedTwinCritic, SharedGaussianActor
from .replay_buffer import ReplayBatch


@dataclass(frozen=True)
class MASACUpdateMetrics:
    """Finite scalar diagnostics returned by one MASAC update."""

    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    q1_mean: float
    q2_mean: float
    target_q_mean: float
    joint_log_probability_mean: float


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite value")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return result


def _finite_probability(value: object, name: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and within [0, 1]")
    result = float(value)
    lower = 0.0 if allow_zero else np.nextafter(0.0, 1.0)
    if not np.isfinite(result) or result < lower or result > 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return result


class ParameterSharingMASAC:
    """Shared actor, centralized twin critics, and one-batch MASAC updates."""

    def __init__(
        self,
        local_observation_dim: int,
        global_state_dim: int,
        num_relays: int,
        action_dim: int = 3,
        hidden_dims: tuple[int, ...] = (256, 256),
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        initial_alpha: float = 0.2,
        target_entropy: float | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.local_observation_dim = _positive_int(local_observation_dim, "local_observation_dim")
        self.global_state_dim = _positive_int(global_state_dim, "global_state_dim")
        self.num_relays = _positive_int(num_relays, "num_relays")
        self.action_dim = _positive_int(action_dim, "action_dim")
        if (
            isinstance(hidden_dims, (str, bytes))
            or not isinstance(hidden_dims, tuple)
            or not hidden_dims
            or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0 for value in hidden_dims)
        ):
            raise ValueError("hidden_dims must be a non-empty tuple of positive integers")
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.gamma = _finite_probability(gamma, "gamma", allow_zero=True)
        self.tau = _finite_probability(tau, "tau", allow_zero=False)
        self.actor_learning_rate = _finite_positive(actor_learning_rate, "actor_learning_rate")
        self.critic_learning_rate = _finite_positive(critic_learning_rate, "critic_learning_rate")
        self.alpha_learning_rate = _finite_positive(alpha_learning_rate, "alpha_learning_rate")
        self.initial_alpha = _finite_positive(initial_alpha, "initial_alpha")
        if target_entropy is not None:
            if isinstance(target_entropy, bool) or not isinstance(target_entropy, Real):
                raise ValueError("target_entropy must be a finite value")
            target_entropy = float(target_entropy)
            if not np.isfinite(target_entropy):
                raise ValueError("target_entropy must be a finite value")
        self.target_entropy = (
            -float(self.num_relays * self.action_dim)
            if target_entropy is None
            else target_entropy
        )

        try:
            self.device = torch.device("cpu" if device is None else device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("device must be a valid torch device") from error
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")

        self.actor = SharedGaussianActor(
            local_observation_dim=self.local_observation_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)
        self.critic = CentralizedTwinCritic(
            global_state_dim=self.global_state_dim,
            num_relays=self.num_relays,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_learning_rate
        )
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(self.initial_alpha), dtype=torch.float32, device=self.device)
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_learning_rate)

    @property
    def alpha(self) -> Tensor:
        """Return the finite positive entropy temperature tensor."""
        return torch.exp(self.log_alpha.clamp(-30.0, 30.0))

    def act(self, local_observations: object, deterministic: bool = False) -> np.ndarray:
        """Select one normalized action per relay without tracking gradients."""
        observations = np.asarray(local_observations, dtype=np.float32)
        if (
            observations.shape != (self.num_relays, self.local_observation_dim)
            or not np.all(np.isfinite(observations))
        ):
            raise ValueError(
                "local_observations must be finite with shape "
                f"({self.num_relays}, {self.local_observation_dim})"
            )
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            actions, _ = self.actor.sample(tensor, deterministic=deterministic)
        result = actions.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=True)
        if not np.all(np.isfinite(result)):
            raise ValueError("actor produced non-finite actions")
        return np.clip(result, -1.0, 1.0).astype(np.float32, copy=False)

    def _prepare_batch(self, batch: ReplayBatch) -> dict[str, Tensor]:
        if not isinstance(batch, ReplayBatch):
            raise ValueError("batch must be a ReplayBatch")
        field_names = (
            "local_observations",
            "global_states",
            "applied_actions",
            "rewards",
            "next_local_observations",
            "next_global_states",
            "terminated",
            "truncated",
        )
        if not all(isinstance(getattr(batch, name), Tensor) for name in field_names):
            raise ValueError("all ReplayBatch fields must be torch.Tensor values")
        tensors = {
            name: getattr(batch, name).to(device=self.device, dtype=torch.float32)
            for name in field_names
        }
        local = tensors["local_observations"]
        global_state = tensors["global_states"]
        actions = tensors["applied_actions"]
        rewards = tensors["rewards"]
        if local.ndim != 3 or local.shape[1:] != (self.num_relays, self.local_observation_dim):
            raise ValueError("local_observations has an incompatible shape")
        batch_size = local.shape[0]
        expected_shapes = {
            "global_states": (batch_size, self.global_state_dim),
            "applied_actions": (batch_size, self.num_relays, self.action_dim),
            "rewards": (batch_size, 1),
            "next_local_observations": (batch_size, self.num_relays, self.local_observation_dim),
            "next_global_states": (batch_size, self.global_state_dim),
            "terminated": (batch_size, 1),
            "truncated": (batch_size, 1),
        }
        for name, shape in expected_shapes.items():
            if tensors[name].shape != shape:
                raise ValueError(f"{name} has an incompatible shape")
        if batch_size <= 0:
            raise ValueError("batch must contain at least one transition")
        if torch.any(actions < -1.0) or torch.any(actions > 1.0):
            raise ValueError("applied_actions must be within [-1, 1]")
        for name in ("terminated", "truncated"):
            if torch.any(tensors[name] < 0.0) or torch.any(tensors[name] > 1.0):
                raise ValueError(f"{name} must be within [0, 1]")
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain only finite values")
        return tensors

    @staticmethod
    def _joint_log_probability(log_probability: Tensor) -> Tensor:
        if log_probability.ndim != 3 or log_probability.shape[-1] != 1:
            raise ValueError("log_probability must have shape (batch, num_relays, 1)")
        return log_probability.sum(dim=1)

    def compute_critic_target(self, batch: ReplayBatch) -> Tensor:
        """Compute SAC targets, masking only true termination transitions."""
        tensors = self._prepare_batch(batch)
        with torch.no_grad():
            next_action, next_log_probability = self.actor.sample(
                tensors["next_local_observations"]
            )
            joint_log_probability = self._joint_log_probability(next_log_probability)
            target_q1, target_q2 = self.target_critic(
                tensors["next_global_states"], next_action
            )
            next_q = torch.minimum(target_q1, target_q2)
            target = tensors["rewards"] + self.gamma * (1.0 - tensors["terminated"]) * (
                next_q - self.alpha.detach() * joint_log_probability
            )
        if not torch.isfinite(target).all():
            raise ValueError("critic target is non-finite")
        return target

    def update(self, batch: ReplayBatch) -> MASACUpdateMetrics:
        """Perform one critic, actor, alpha, and Polyak target update."""
        tensors = self._prepare_batch(batch)
        target = self.compute_critic_target(batch).detach()

        q1, q2 = self.critic(tensors["global_states"], tensors["applied_actions"])
        critic_loss_tensor = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss_tensor.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        try:
            actions, log_probability = self.actor.sample(tensors["local_observations"])
            joint_log_probability = self._joint_log_probability(log_probability)
            actor_q1, actor_q2 = self.critic(tensors["global_states"], actions)
            actor_loss_tensor = (
                self.alpha.detach() * joint_log_probability - torch.minimum(actor_q1, actor_q2)
            ).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss_tensor.backward()
            self.actor_optimizer.step()
        finally:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)

        alpha_loss_tensor = -(
            self.log_alpha * (joint_log_probability.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss_tensor.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, online_parameter in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_parameter.mul_(1.0 - self.tau).add_(online_parameter, alpha=self.tau)

        values = {
            "critic_loss": critic_loss_tensor.detach(),
            "actor_loss": actor_loss_tensor.detach(),
            "alpha_loss": alpha_loss_tensor.detach(),
            "alpha": self.alpha.detach(),
            "q1_mean": q1.detach().mean(),
            "q2_mean": q2.detach().mean(),
            "target_q_mean": target.detach().mean(),
            "joint_log_probability_mean": joint_log_probability.detach().mean(),
        }
        if not all(torch.isfinite(value).item() for value in values.values()):
            raise ValueError("MASAC update produced non-finite metrics")
        return MASACUpdateMetrics(**{name: float(value.cpu()) for name, value in values.items()})


__all__ = ["MASACUpdateMetrics", "ParameterSharingMASAC"]
