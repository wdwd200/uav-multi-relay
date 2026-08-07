"""Independent atomic MAPPO checkpoint persistence."""
from __future__ import annotations
import os, tempfile
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
import torch
from ..learning import MAPPOAgent, MAPPOConfig

@dataclass(frozen=True)
class MAPPOCheckpointMetadata:
    environment_steps: int
    updates: int
    completed_episodes: int
    def __post_init__(self) -> None:
        for name in ("environment_steps", "updates", "completed_episodes"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), Integral) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")

def save_mappo_checkpoint(path: str | Path, agent: MAPPOAgent, metadata: MAPPOCheckpointMetadata) -> Path:
    if not isinstance(agent, MAPPOAgent) or not isinstance(metadata, MAPPOCheckpointMetadata): raise ValueError("agent and metadata have incompatible types")
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "agent_config": {"local_observation_dim":agent.local_observation_dim,"global_state_dim":agent.global_state_dim,"num_relays":agent.num_relays,"action_dim":agent.action_dim,"hidden_dims":agent.hidden_dims,"config":asdict(agent.config)}, "actor_state_dict":agent.actor.state_dict(), "value_critic_state_dict":agent.value_critic.state_dict(), "actor_optimizer_state_dict":agent.actor_optimizer.state_dict(), "critic_optimizer_state_dict":agent.critic_optimizer.state_dict(), "metadata":asdict(metadata)}
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as handle: temporary=Path(handle.name)
        torch.save(payload, temporary); os.replace(temporary, destination)
    except Exception:
        if temporary is not None: temporary.unlink(missing_ok=True)
        raise
    return destination

def load_mappo_checkpoint(path: str | Path, device: str | torch.device | None = None) -> tuple[MAPPOAgent, MAPPOCheckpointMetadata]:
    source=Path(path)
    if not source.is_file(): raise ValueError("checkpoint path does not exist")
    target=torch.device("cpu" if device is None else device)
    try:
        try: payload=torch.load(source,map_location=target,weights_only=True)
        except (TypeError,RuntimeError): payload=torch.load(source,map_location=target)
    except Exception as error: raise ValueError("unable to load MAPPO checkpoint") from error
    required={"format_version","agent_config","actor_state_dict","value_critic_state_dict","actor_optimizer_state_dict","critic_optimizer_state_dict","metadata"}
    if not isinstance(payload,dict) or set(payload)!=required or payload["format_version"]!=1: raise ValueError("unsupported or incomplete MAPPO checkpoint")
    config=payload["agent_config"]
    if not isinstance(config,dict) or set(config)!={"local_observation_dim","global_state_dim","num_relays","action_dim","hidden_dims","config"}: raise ValueError("checkpoint agent configuration is malformed")
    try:
        agent=MAPPOAgent(config["local_observation_dim"],config["global_state_dim"],config["num_relays"],config["action_dim"],tuple(config["hidden_dims"]),MAPPOConfig(**config["config"]),target)
        metadata=MAPPOCheckpointMetadata(**payload["metadata"]); agent.actor.load_state_dict(payload["actor_state_dict"]); agent.value_critic.load_state_dict(payload["value_critic_state_dict"]); agent.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"]); agent.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
    except Exception as error: raise ValueError("MAPPO checkpoint contents are incompatible") from error
    return agent, metadata

__all__=["MAPPOCheckpointMetadata","load_mappo_checkpoint","save_mappo_checkpoint"]
