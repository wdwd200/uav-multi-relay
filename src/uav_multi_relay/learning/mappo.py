"""Parameter-sharing MAPPO primitives with requested-action PPO semantics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from .networks import CentralizedValueCritic, SharedGaussianActor


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}value")
    return result


@dataclass(frozen=True)
class MAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    update_epochs: int = 10
    mini_batch_size: int = 256
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5

    def __post_init__(self) -> None:
        gamma = _finite(self.gamma, "gamma")
        gae_lambda = _finite(self.gae_lambda, "gae_lambda")
        clip_ratio = _finite(self.clip_ratio, "clip_ratio", positive=True)
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0 or clip_ratio > 1.0:
            raise ValueError("gamma, gae_lambda, and clip_ratio are out of range")
        for name in ("update_epochs", "mini_batch_size"):
            _positive_int(getattr(self, name), name)
        for name in ("actor_learning_rate", "critic_learning_rate", "max_gradient_norm"):
            _finite(getattr(self, name), name, positive=True)
        for name in ("value_loss_coefficient", "entropy_coefficient"):
            if _finite(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class MAPPOUpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    actor_gradient_norm: float
    critic_gradient_norm: float
    value_mean: float
    return_mean: float
    advantage_mean: float
    advantage_std: float
    requested_applied_mismatch_mean: float
    requested_applied_mismatch_rate: float


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray,
    terminated: np.ndarray, truncated: np.ndarray, gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE with terminated bootstrap masking and truncated trace boundaries."""
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in (rewards, values, next_values, terminated, truncated)]
    if not arrays[0].size or any(array.shape != arrays[0].shape or not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("GAE arrays must be non-empty, finite, and have equal shape")
    gamma = _finite(gamma, "gamma")
    gae_lambda = _finite(gae_lambda, "gae_lambda")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be within [0, 1]")
    rewards, values, next_values, terminated, truncated = arrays
    if np.any((terminated < 0) | (terminated > 1)) or np.any((truncated < 0) | (truncated > 1)):
        raise ValueError("termination flags must be within [0, 1]")
    advantages = np.zeros_like(rewards)
    accumulator = 0.0
    for index in range(rewards.size - 1, -1, -1):
        bootstrap_mask = 1.0 - terminated[index]
        trace_mask = 1.0 - max(terminated[index], truncated[index])
        delta = rewards[index] + gamma * bootstrap_mask * next_values[index] - values[index]
        accumulator = delta + gamma * gae_lambda * trace_mask * accumulator
        advantages[index] = accumulator
    returns = advantages + values
    if not np.all(np.isfinite(advantages)) or not np.all(np.isfinite(returns)):
        raise ValueError("GAE produced non-finite values")
    return advantages.astype(np.float32), returns.astype(np.float32)


class MAPPORollout:
    """Fixed-capacity on-policy rollout; applied actions remain diagnostics only."""

    def __init__(self, capacity: int, num_relays: int, local_observation_dim: int, global_state_dim: int, action_dim: int = 3) -> None:
        self.capacity = _positive_int(capacity, "capacity")
        self.num_relays = _positive_int(num_relays, "num_relays")
        self.local_observation_dim = _positive_int(local_observation_dim, "local_observation_dim")
        self.global_state_dim = _positive_int(global_state_dim, "global_state_dim")
        self.action_dim = _positive_int(action_dim, "action_dim")
        self.clear()

    def clear(self) -> None:
        self.local_observations: list[np.ndarray] = []
        self.global_states: list[np.ndarray] = []
        self.requested_actions: list[np.ndarray] = []
        self.applied_actions: list[np.ndarray] = []
        self.old_joint_log_probabilities: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.next_values: list[float] = []
        self.terminated: list[float] = []
        self.truncated: list[float] = []

    @property
    def size(self) -> int:
        return len(self.rewards)

    @property
    def is_full(self) -> bool:
        return self.size == self.capacity

    def add(self, local_observation: object, global_state: object, requested_action: object, applied_action: object, old_joint_log_probability: object, reward: object, value: object, next_value: object, terminated: object, truncated: object) -> None:
        if self.is_full:
            raise ValueError("rollout is already full")
        local = np.asarray(local_observation, dtype=np.float32)
        state = np.asarray(global_state, dtype=np.float32)
        requested = np.asarray(requested_action, dtype=np.float32)
        applied = np.asarray(applied_action, dtype=np.float32)
        expected = ((self.num_relays, self.local_observation_dim), (self.global_state_dim,), (self.num_relays, self.action_dim), (self.num_relays, self.action_dim))
        if (local.shape, state.shape, requested.shape, applied.shape) != expected:
            raise ValueError("rollout transition has incompatible shape")
        scalars = [float(old_joint_log_probability), float(reward), float(value), float(next_value), float(terminated), float(truncated)]
        if not all(np.isfinite(item) for item in scalars) or not all(np.all(np.isfinite(item)) for item in (local, state, requested, applied)):
            raise ValueError("rollout transition must be finite")
        if np.any(np.abs(requested) > 1.0) or np.any(np.abs(applied) > 1.0) or scalars[-2] not in (0.0, 1.0) or scalars[-1] not in (0.0, 1.0):
            raise ValueError("rollout actions or flags are invalid")
        self.local_observations.append(local.copy()); self.global_states.append(state.copy())
        self.requested_actions.append(requested.copy()); self.applied_actions.append(applied.copy())
        self.old_joint_log_probabilities.append(scalars[0]); self.rewards.append(scalars[1]); self.values.append(scalars[2]); self.next_values.append(scalars[3]); self.terminated.append(scalars[4]); self.truncated.append(scalars[5])

    def arrays(self, gamma: float, gae_lambda: float) -> dict[str, np.ndarray]:
        if not self.is_full:
            raise ValueError("PPO update requires a full rollout")
        advantages, returns = compute_gae(np.asarray(self.rewards), np.asarray(self.values), np.asarray(self.next_values), np.asarray(self.terminated), np.asarray(self.truncated), gamma, gae_lambda)
        result = {
            "local_observations": np.stack(self.local_observations), "global_states": np.stack(self.global_states),
            "requested_actions": np.stack(self.requested_actions), "applied_actions": np.stack(self.applied_actions),
            "old_joint_log_probabilities": np.asarray(self.old_joint_log_probabilities, dtype=np.float32)[:, None],
            "rewards": np.asarray(self.rewards, dtype=np.float32)[:, None], "values": np.asarray(self.values, dtype=np.float32)[:, None],
            "next_values": np.asarray(self.next_values, dtype=np.float32)[:, None], "terminated": np.asarray(self.terminated, dtype=np.float32)[:, None], "truncated": np.asarray(self.truncated, dtype=np.float32)[:, None],
            "advantages": advantages[:, None], "returns": returns[:, None],
        }
        if not all(np.all(np.isfinite(value)) for value in result.values()):
            raise ValueError("rollout arrays are non-finite")
        return result


class MAPPOAgent:
    """Shared Gaussian actor, centralized value critic, and PPO update."""

    def __init__(self, local_observation_dim: int, global_state_dim: int, num_relays: int, action_dim: int = 3, hidden_dims: tuple[int, ...] = (256, 256), config: MAPPOConfig | None = None, device: str | torch.device | None = None) -> None:
        self.local_observation_dim = _positive_int(local_observation_dim, "local_observation_dim")
        self.global_state_dim = _positive_int(global_state_dim, "global_state_dim")
        self.num_relays = _positive_int(num_relays, "num_relays")
        self.action_dim = _positive_int(action_dim, "action_dim")
        if not isinstance(hidden_dims, tuple) or not hidden_dims or any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in hidden_dims):
            raise ValueError("hidden_dims must be a non-empty tuple of positive integers")
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        self.config = MAPPOConfig() if config is None else config
        if not isinstance(self.config, MAPPOConfig):
            raise ValueError("config must be a MAPPOConfig")
        try:
            self.device = torch.device("cpu" if device is None else device)
        except Exception as error:
            raise ValueError("device must be valid") from error
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but unavailable")
        self.actor = SharedGaussianActor(self.local_observation_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.value_critic = CentralizedValueCritic(self.global_state_dim, self.hidden_dims).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.value_critic.parameters(), lr=self.config.critic_learning_rate)

    @staticmethod
    def _gradient_norm(parameters: object) -> Tensor:
        grads = [parameter.grad.detach().reshape(-1) for parameter in parameters if parameter.grad is not None]
        result = torch.linalg.vector_norm(torch.cat(grads)) if grads else torch.zeros(())
        if not torch.isfinite(result):
            raise ValueError("gradient norm is non-finite")
        return result

    def act_with_stats(self, local_observations: np.ndarray, global_state: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        local = np.asarray(local_observations, dtype=np.float32); state = np.asarray(global_state, dtype=np.float32)
        if local.shape != (self.num_relays, self.local_observation_dim) or state.shape != (self.global_state_dim,) or not np.all(np.isfinite(local)) or not np.all(np.isfinite(state)):
            raise ValueError("observations are incompatible")
        with torch.no_grad():
            actions, per_relay = self.actor.sample(torch.as_tensor(local, device=self.device).unsqueeze(0), deterministic=deterministic)
            value = self.value_critic(torch.as_tensor(state, device=self.device).unsqueeze(0))
        result = actions.squeeze(0).cpu().numpy().astype(np.float32)
        joint = float(per_relay.sum(dim=1).item()); value_result = float(value.item())
        if not np.all(np.isfinite(result)) or not np.isfinite(joint) or not np.isfinite(value_result):
            raise ValueError("MAPPO action statistics are non-finite")
        return result, joint, value_result

    def act(self, local_observations: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # act is intentionally actor-only; callers needing value use act_with_stats.
        local = np.asarray(local_observations, dtype=np.float32)
        if local.shape != (self.num_relays, self.local_observation_dim):
            raise ValueError("local observations are incompatible")
        with torch.no_grad():
            action, _ = self.actor.sample(torch.as_tensor(local, device=self.device).unsqueeze(0), deterministic=deterministic)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def value(self, global_state: np.ndarray) -> float:
        state = np.asarray(global_state, dtype=np.float32)
        if state.shape != (self.global_state_dim,) or not np.all(np.isfinite(state)):
            raise ValueError("global state is incompatible")
        with torch.no_grad():
            value = self.value_critic(torch.as_tensor(state, device=self.device).unsqueeze(0))
        return float(value.item())

    def update(self, rollout: MAPPORollout) -> MAPPOUpdateMetrics:
        if not isinstance(rollout, MAPPORollout):
            raise ValueError("rollout must be a MAPPORollout")
        arrays = rollout.arrays(self.config.gamma, self.config.gae_lambda)
        tensors = {name: torch.as_tensor(value, dtype=torch.float32, device=self.device) for name, value in arrays.items()}
        advantages = tensors["advantages"]
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        if not torch.isfinite(normalized_advantages).all():
            raise ValueError("normalized advantages are non-finite")
        mismatch = torch.linalg.vector_norm(tensors["requested_actions"] - tensors["applied_actions"], dim=-1)
        scalar_metrics: dict[str, list[Tensor]] = {name: [] for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "actor_gradient_norm", "critic_gradient_norm", "value_mean")}
        size = rollout.size
        for _ in range(self.config.update_epochs):
            for indices in torch.randperm(size, device=self.device).split(self.config.mini_batch_size):
                new_log_probability, _, entropy = self.actor.evaluate_actions(tensors["local_observations"][indices], tensors["requested_actions"][indices])
                ratio = torch.exp(new_log_probability - tensors["old_joint_log_probabilities"][indices])
                unclipped = ratio * normalized_advantages[indices]
                clipped = ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * normalized_advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean() - self.config.entropy_coefficient * entropy.mean()
                self.actor_optimizer.zero_grad(set_to_none=True); policy_loss.backward()
                actor_norm = self._gradient_norm(self.actor.parameters()).detach(); clip_grad_norm_(self.actor.parameters(), self.config.max_gradient_norm); self.actor_optimizer.step()
                values = self.value_critic(tensors["global_states"][indices])
                value_loss = F.mse_loss(values, tensors["returns"][indices])
                self.critic_optimizer.zero_grad(set_to_none=True); (self.config.value_loss_coefficient * value_loss).backward()
                critic_norm = self._gradient_norm(self.value_critic.parameters()).detach(); clip_grad_norm_(self.value_critic.parameters(), self.config.max_gradient_norm); self.critic_optimizer.step()
                scalar_metrics["policy_loss"].append(policy_loss.detach()); scalar_metrics["value_loss"].append(value_loss.detach()); scalar_metrics["entropy"].append(entropy.mean().detach())
                scalar_metrics["approx_kl"].append((tensors["old_joint_log_probabilities"][indices] - new_log_probability.detach()).mean()); scalar_metrics["clip_fraction"].append((torch.abs(ratio.detach() - 1.0) > self.config.clip_ratio).float().mean())
                scalar_metrics["actor_gradient_norm"].append(actor_norm); scalar_metrics["critic_gradient_norm"].append(critic_norm); scalar_metrics["value_mean"].append(values.detach().mean())
        values = {name: torch.stack(items).mean() for name, items in scalar_metrics.items()}
        values.update({"return_mean": tensors["returns"].mean(), "advantage_mean": advantages.mean(), "advantage_std": advantages.std(unbiased=False), "requested_applied_mismatch_mean": mismatch.mean(), "requested_applied_mismatch_rate": mismatch.gt(1e-6).any(dim=1).float().mean()})
        if not all(torch.isfinite(value).item() for value in values.values()):
            raise ValueError("MAPPO update produced non-finite metrics")
        return MAPPOUpdateMetrics(**{name: float(value.cpu()) for name, value in values.items()})


__all__ = ["MAPPOAgent", "MAPPOConfig", "MAPPOUpdateMetrics", "MAPPORollout", "compute_gae"]
