"""Atomic checkpoints shared by MATD3 and MADDPG."""
from __future__ import annotations
import os,tempfile
from dataclasses import asdict,dataclass
from numbers import Integral
from pathlib import Path
import torch
from ..learning import ParameterSharingMADDPG,ParameterSharingMATD3
DeterministicAgent=ParameterSharingMADDPG|ParameterSharingMATD3
@dataclass(frozen=True)
class DeterministicCheckpointMetadata:
    environment_steps:int; updates:int; completed_episodes:int
    def __post_init__(self)->None:
        for name in ("environment_steps","updates","completed_episodes"):
            if isinstance(getattr(self,name),bool) or not isinstance(getattr(self,name),Integral) or getattr(self,name)<0: raise ValueError(f"{name} must be a non-negative integer")
def _config(agent:DeterministicAgent)->dict[str,object]:
    names=("local_observation_dim","global_state_dim","num_relays","action_dim","hidden_dims","gamma","tau","actor_learning_rate","critic_learning_rate")
    config={x:getattr(agent,x) for x in names}
    if agent.algorithm=="matd3": config|={x:getattr(agent,x) for x in ("policy_noise_std","noise_clip","policy_delay")}
    return config
def save_deterministic_checkpoint(path:str|Path,agent:DeterministicAgent,metadata:DeterministicCheckpointMetadata)->Path:
    if not isinstance(agent,(ParameterSharingMADDPG,ParameterSharingMATD3)) or not isinstance(metadata,DeterministicCheckpointMetadata): raise ValueError("agent and metadata have incompatible types")
    destination=Path(path); destination.parent.mkdir(parents=True,exist_ok=True)
    payload={"format_version":1,"algorithm":agent.algorithm,"agent_config":_config(agent),"actor_state_dict":agent.actor.state_dict(),"target_actor_state_dict":agent.target_actor.state_dict(),"critic_state_dict":agent.critic.state_dict(),"target_critic_state_dict":agent.target_critic.state_dict(),"actor_optimizer_state_dict":agent.actor_optimizer.state_dict(),"critic_optimizer_state_dict":agent.critic_optimizer.state_dict(),"update_count":getattr(agent,"update_count",0),"metadata":asdict(metadata)}
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.",suffix=".tmp",dir=destination.parent,delete=False) as handle: temporary=Path(handle.name)
        torch.save(payload,temporary);os.replace(temporary,destination)
    except Exception:
        if temporary is not None: temporary.unlink(missing_ok=True)
        raise
    return destination
def load_deterministic_checkpoint(path:str|Path,*,algorithm:str|None=None,device:str|torch.device|None=None)->tuple[DeterministicAgent,DeterministicCheckpointMetadata]:
    source=Path(path)
    if not source.is_file(): raise ValueError("checkpoint path does not exist")
    target=torch.device("cpu" if device is None else device)
    try:
        try: payload=torch.load(source,map_location=target,weights_only=True)
        except (TypeError,RuntimeError): payload=torch.load(source,map_location=target)
    except Exception as error: raise ValueError("unable to load deterministic checkpoint") from error
    required={"format_version","algorithm","agent_config","actor_state_dict","target_actor_state_dict","critic_state_dict","target_critic_state_dict","actor_optimizer_state_dict","critic_optimizer_state_dict","update_count","metadata"}
    if not isinstance(payload,dict) or set(payload)!=required or payload["format_version"]!=1 or payload["algorithm"] not in {"matd3","maddpg"}: raise ValueError("unsupported or incomplete deterministic checkpoint")
    if algorithm is not None and payload["algorithm"]!=algorithm: raise ValueError("checkpoint algorithm type does not match requested algorithm")
    try:
        cls=ParameterSharingMATD3 if payload["algorithm"]=="matd3" else ParameterSharingMADDPG; agent=cls(device=target,**payload["agent_config"]); metadata=DeterministicCheckpointMetadata(**payload["metadata"]); agent.actor.load_state_dict(payload["actor_state_dict"]);agent.target_actor.load_state_dict(payload["target_actor_state_dict"]);agent.critic.load_state_dict(payload["critic_state_dict"]);agent.target_critic.load_state_dict(payload["target_critic_state_dict"]);agent.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"]);agent.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"]);agent.update_count=int(payload["update_count"])
    except Exception as error: raise ValueError("deterministic checkpoint contents are incompatible") from error
    for module in (agent.target_actor,agent.target_critic):
        for p in module.parameters(): p.requires_grad_(False)
    return agent,metadata
def load_matd3_checkpoint(path:str|Path,device:str|torch.device|None=None): return load_deterministic_checkpoint(path,algorithm="matd3",device=device)
def load_maddpg_checkpoint(path:str|Path,device:str|torch.device|None=None): return load_deterministic_checkpoint(path,algorithm="maddpg",device=device)
__all__=["DeterministicCheckpointMetadata","save_deterministic_checkpoint","load_deterministic_checkpoint","load_matd3_checkpoint","load_maddpg_checkpoint"]
