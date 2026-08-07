"""Portable, atomic MASAC checkpoint persistence."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from ..learning import ParameterSharingMASAC


@dataclass(frozen=True)
class MASACCheckpointMetadata:
    environment_steps: int
    updates: int
    completed_episodes: int

    def __post_init__(self) -> None:
        for name in ("environment_steps", "updates", "completed_episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


def _agent_config(agent: ParameterSharingMASAC) -> dict[str, object]:
    names = (
        "local_observation_dim", "global_state_dim", "num_relays", "action_dim",
        "hidden_dims", "gamma", "tau", "actor_learning_rate", "critic_learning_rate",
        "alpha_learning_rate", "initial_alpha", "target_entropy", "critic_gradient_clip_norm",
    )
    try:
        config = {name: getattr(agent, name) for name in names}
    except AttributeError as error:
        raise ValueError("agent is missing checkpoint configuration") from error
    config["hidden_dims"] = tuple(int(value) for value in config["hidden_dims"])
    if not all(np.isfinite(float(config[name])) for name in names[5:-1]):
        raise ValueError("agent configuration must be finite")
    if config["critic_gradient_clip_norm"] is not None and not np.isfinite(float(config["critic_gradient_clip_norm"])):
        raise ValueError("agent configuration must be finite")
    return config


def save_masac_checkpoint(
    path: str | Path,
    agent: ParameterSharingMASAC,
    metadata: MASACCheckpointMetadata,
) -> Path:
    if not isinstance(agent, ParameterSharingMASAC) or not isinstance(metadata, MASACCheckpointMetadata):
        raise ValueError("agent and metadata have incompatible types")
    destination = Path(path)
    if destination.name == "":
        raise ValueError("path must name a checkpoint file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "agent_config": _agent_config(agent),
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "target_critic_state_dict": agent.target_critic.state_dict(),
        "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
        "alpha_optimizer_state_dict": agent.alpha_optimizer.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "metadata": {
            "environment_steps": int(metadata.environment_steps),
            "updates": int(metadata.updates),
            "completed_episodes": int(metadata.completed_episodes),
        },
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def _load_payload(path: Path, device: torch.device) -> dict[str, object]:
    try:
        try:
            payload = torch.load(path, map_location=device, weights_only=True)
        except (TypeError, RuntimeError):
            payload = torch.load(path, map_location=device)
    except Exception as error:
        raise ValueError("unable to load checkpoint") from error
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    required = {
        "format_version", "agent_config", "actor_state_dict", "critic_state_dict",
        "target_critic_state_dict", "actor_optimizer_state_dict", "critic_optimizer_state_dict",
        "alpha_optimizer_state_dict", "log_alpha", "metadata",
    }
    if set(payload) != required or payload["format_version"] != 1:
        raise ValueError("unsupported or incomplete checkpoint")
    return payload


def load_masac_checkpoint(
    path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[ParameterSharingMASAC, MASACCheckpointMetadata]:
    source = Path(path)
    if not source.is_file():
        raise ValueError("checkpoint path does not exist")
    try:
        target_device = torch.device("cpu" if device is None else device)
    except Exception as error:
        raise ValueError("device must be valid") from error
    payload = _load_payload(source, target_device)
    config = payload["agent_config"]
    metadata_data = payload["metadata"]
    if not isinstance(config, dict) or not isinstance(metadata_data, dict):
        raise ValueError("checkpoint configuration is malformed")
    expected = {"local_observation_dim", "global_state_dim", "num_relays", "action_dim", "hidden_dims", "gamma", "tau", "actor_learning_rate", "critic_learning_rate", "alpha_learning_rate", "initial_alpha", "target_entropy"}
    current_expected = expected | {"critic_gradient_clip_norm"}
    if set(config) not in (expected, current_expected):
        raise ValueError("checkpoint agent configuration is incomplete")
    try:
        config = dict(config)
        config.setdefault("critic_gradient_clip_norm", None)
        agent = ParameterSharingMASAC(device=target_device, **config)
        metadata = MASACCheckpointMetadata(**metadata_data)
        agent.actor.load_state_dict(payload["actor_state_dict"])
        agent.critic.load_state_dict(payload["critic_state_dict"])
        agent.target_critic.load_state_dict(payload["target_critic_state_dict"])
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        agent.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        agent.alpha_optimizer.load_state_dict(payload["alpha_optimizer_state_dict"])
        log_alpha = payload["log_alpha"]
        if not isinstance(log_alpha, torch.Tensor) or log_alpha.shape != () or not torch.isfinite(log_alpha).item():
            raise ValueError("log_alpha is malformed")
        agent.log_alpha.data.copy_(log_alpha.to(agent.device))
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("checkpoint contents are incompatible") from error
    for parameter in agent.target_critic.parameters():
        parameter.requires_grad_(False)
    return agent, metadata


__all__ = ["MASACCheckpointMetadata", "load_masac_checkpoint", "save_masac_checkpoint"]
