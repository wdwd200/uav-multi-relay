"""Shared implementation primitives for parameter-sharing deterministic MARL."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Real
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .networks import CentralizedCritic, CentralizedTwinCritic, SharedDeterministicActor
from .replay_buffer import ReplayBatch


@dataclass(frozen=True)
class DeterministicUpdateMetrics:
    critic_loss: float
    actor_loss: float
    current_q_mean: float
    target_q_mean: float
    td_error_mean: float
    actor_gradient_norm: float
    critic_gradient_norm: float
    actor_updated: float
    policy_delay_counter: float = 0.0
    target_noise_mean: float = 0.0
    target_noise_std: float = 0.0
    target_noise_max: float = 0.0


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0: raise ValueError(f"{name} must be positive")
    return int(value)

def _probability(value: object, name: str, *, zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)) or float(value) < (0 if zero else np.nextafter(0., 1.)) or float(value) > 1: raise ValueError(f"{name} must lie in [0, 1]")
    return float(value)

def _learning_rate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)) or float(value) <= 0: raise ValueError(f"{name} must be positive and finite")
    return float(value)

class _DeterministicAgent:
    algorithm = "deterministic"
    def __init__(self, local_observation_dim: int, global_state_dim: int, num_relays: int, action_dim: int = 3,
                 hidden_dims: tuple[int, ...] = (256, 256), gamma: float = .99, tau: float = .005,
                 actor_learning_rate: float = 3e-4, critic_learning_rate: float = 3e-4,
                 device: str | torch.device | None = None) -> None:
        self.local_observation_dim=_positive(local_observation_dim,"local_observation_dim"); self.global_state_dim=_positive(global_state_dim,"global_state_dim"); self.num_relays=_positive(num_relays,"num_relays"); self.action_dim=_positive(action_dim,"action_dim")
        if not isinstance(hidden_dims,tuple) or not hidden_dims or any(not isinstance(x,int) or x<=0 for x in hidden_dims): raise ValueError("hidden_dims must be a non-empty tuple of positive integers")
        self.hidden_dims=tuple(hidden_dims); self.gamma=_probability(gamma,"gamma"); self.tau=_probability(tau,"tau",zero=False); self.actor_learning_rate=_learning_rate(actor_learning_rate,"actor_learning_rate"); self.critic_learning_rate=_learning_rate(critic_learning_rate,"critic_learning_rate")
        self.device=torch.device("cpu" if device is None else device)
        self.actor=SharedDeterministicActor(self.local_observation_dim,self.action_dim,self.hidden_dims).to(self.device); self.target_actor=copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=self.actor_learning_rate)
        for p in self.target_actor.parameters(): p.requires_grad_(False)

    def act(self, local_observations: object, deterministic: bool = True) -> np.ndarray:
        observations=np.asarray(local_observations,dtype=np.float32)
        if observations.shape != (self.num_relays,self.local_observation_dim) or not np.all(np.isfinite(observations)): raise ValueError("local_observations has incompatible shape or non-finite values")
        with torch.no_grad(): actions=self.actor(torch.as_tensor(observations,device=self.device).unsqueeze(0))
        result=actions.squeeze(0).cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(result)): raise ValueError("actor produced non-finite actions")
        return result

    def _prepare_batch(self,batch: ReplayBatch)->dict[str,Tensor]:
        if not isinstance(batch,ReplayBatch): raise ValueError("batch must be a ReplayBatch")
        names=("local_observations","global_states","applied_actions","rewards","next_local_observations","next_global_states","terminated","truncated")
        values={name:getattr(batch,name).to(self.device,dtype=torch.float32) for name in names}
        n=values["local_observations"].shape[0]
        shapes={"local_observations":(n,self.num_relays,self.local_observation_dim),"global_states":(n,self.global_state_dim),"applied_actions":(n,self.num_relays,self.action_dim),"rewards":(n,1),"next_local_observations":(n,self.num_relays,self.local_observation_dim),"next_global_states":(n,self.global_state_dim),"terminated":(n,1),"truncated":(n,1)}
        if n<=0 or any(values[k].shape != shape for k,shape in shapes.items()) or any(not torch.isfinite(x).all() for x in values.values()): raise ValueError("batch has incompatible shape or non-finite values")
        if torch.any(values["applied_actions"] < -1) or torch.any(values["applied_actions"] > 1): raise ValueError("applied_actions must lie in [-1, 1]")
        return values

    @staticmethod
    def _gradient_norm(parameters: object) -> Tensor:
        gradients=[p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
        return torch.linalg.vector_norm(torch.cat(gradients)) if gradients else torch.zeros((),dtype=torch.float32)
    def _soft_update(self, target: torch.nn.Module, online: torch.nn.Module) -> None:
        with torch.no_grad():
            for t,p in zip(target.parameters(),online.parameters()): t.mul_(1-self.tau).add_(p,alpha=self.tau)
    @staticmethod
    def _metrics(**kwargs: Tensor | float) -> DeterministicUpdateMetrics:
        values={name:float(value.detach().cpu()) if isinstance(value,Tensor) else float(value) for name,value in kwargs.items()}
        if not all(np.isfinite(v) for v in values.values()): raise ValueError("deterministic update produced non-finite metrics")
        return DeterministicUpdateMetrics(**values)


class ParameterSharingMADDPG(_DeterministicAgent):
    algorithm="maddpg"
    def __init__(self,*args: object,**kwargs: object)->None:
        super().__init__(*args,**kwargs); self.critic=CentralizedCritic(self.global_state_dim,self.num_relays,self.action_dim,self.hidden_dims).to(self.device); self.target_critic=copy.deepcopy(self.critic).to(self.device); self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=self.critic_learning_rate)
        for p in self.target_critic.parameters(): p.requires_grad_(False)
    def compute_target_q(self,batch: ReplayBatch)->Tensor:
        x=self._prepare_batch(batch)
        with torch.no_grad(): q=self.target_critic(x["next_global_states"],self.target_actor(x["next_local_observations"])); target=x["rewards"]+self.gamma*(1-x["terminated"])*q
        if not torch.isfinite(target).all(): raise ValueError("target Q is non-finite")
        return target
    def update(self,batch: ReplayBatch)->DeterministicUpdateMetrics:
        x=self._prepare_batch(batch); target=self.compute_target_q(batch).detach(); current=self.critic(x["global_states"],x["applied_actions"]); loss=F.mse_loss(current,target); self.critic_optimizer.zero_grad(set_to_none=True); loss.backward(); critic_norm=self._gradient_norm(self.critic.parameters()); self.critic_optimizer.step()
        for p in self.critic.parameters(): p.requires_grad_(False)
        try:
            actor_q=self.critic(x["global_states"],self.actor(x["local_observations"])); actor_loss=-actor_q.mean(); self.actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward(); actor_norm=self._gradient_norm(self.actor.parameters()); self.actor_optimizer.step()
        finally:
            for p in self.critic.parameters(): p.requires_grad_(True)
        self._soft_update(self.target_actor,self.actor); self._soft_update(self.target_critic,self.critic)
        return self._metrics(critic_loss=loss,actor_loss=actor_loss,current_q_mean=current.mean(),target_q_mean=target.mean(),td_error_mean=torch.abs(current-target).mean(),actor_gradient_norm=actor_norm,critic_gradient_norm=critic_norm,actor_updated=1.)


class ParameterSharingMATD3(_DeterministicAgent):
    algorithm="matd3"
    def __init__(self,*args: object,policy_noise_std: float=.2,noise_clip: float=.5,policy_delay: int=2,**kwargs: object)->None:
        super().__init__(*args,**kwargs); self.policy_noise_std=_learning_rate(policy_noise_std,"policy_noise_std"); self.noise_clip=_learning_rate(noise_clip,"noise_clip"); self.policy_delay=_positive(policy_delay,"policy_delay"); self.update_count=0; self.critic=CentralizedTwinCritic(self.global_state_dim,self.num_relays,self.action_dim,self.hidden_dims).to(self.device); self.target_critic=copy.deepcopy(self.critic).to(self.device); self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=self.critic_learning_rate)
        for p in self.target_critic.parameters(): p.requires_grad_(False)
    def _target_action(self,next_local: Tensor)->tuple[Tensor,Tensor]:
        noise=torch.randn_like(self.target_actor(next_local))*self.policy_noise_std; noise=noise.clamp(-self.noise_clip,self.noise_clip); return (self.target_actor(next_local)+noise).clamp(-1.,1.),noise
    def compute_target_q(self,batch: ReplayBatch)->Tensor:
        x=self._prepare_batch(batch)
        with torch.no_grad(): action,_=self._target_action(x["next_local_observations"]); q1,q2=self.target_critic(x["next_global_states"],action); target=x["rewards"]+self.gamma*(1-x["terminated"])*torch.minimum(q1,q2)
        if not torch.isfinite(target).all(): raise ValueError("target Q is non-finite")
        return target
    def update(self,batch: ReplayBatch)->DeterministicUpdateMetrics:
        x=self._prepare_batch(batch)
        with torch.no_grad(): target_action,noise=self._target_action(x["next_local_observations"]); q1t,q2t=self.target_critic(x["next_global_states"],target_action); target=x["rewards"]+self.gamma*(1-x["terminated"])*torch.minimum(q1t,q2t)
        q1,q2=self.critic(x["global_states"],x["applied_actions"]); loss=F.mse_loss(q1,target)+F.mse_loss(q2,target); self.critic_optimizer.zero_grad(set_to_none=True); loss.backward(); critic_norm=self._gradient_norm(self.critic.parameters()); self.critic_optimizer.step(); self.update_count+=1; updated=self.update_count % self.policy_delay == 0; actor_loss=torch.zeros((),device=self.device); actor_norm=torch.zeros((),device=self.device)
        if updated:
            for p in self.critic.parameters(): p.requires_grad_(False)
            try:
                actor_loss=-self.critic(x["global_states"],self.actor(x["local_observations"]))[0].mean(); self.actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward(); actor_norm=self._gradient_norm(self.actor.parameters()); self.actor_optimizer.step()
            finally:
                for p in self.critic.parameters(): p.requires_grad_(True)
            self._soft_update(self.target_actor,self.actor); self._soft_update(self.target_critic,self.critic)
        return self._metrics(critic_loss=loss,actor_loss=actor_loss,current_q_mean=torch.minimum(q1,q2).mean(),target_q_mean=target.mean(),td_error_mean=torch.cat((torch.abs(q1-target),torch.abs(q2-target))).mean(),actor_gradient_norm=actor_norm,critic_gradient_norm=critic_norm,actor_updated=float(updated),policy_delay_counter=float(self.update_count % self.policy_delay),target_noise_mean=noise.mean(),target_noise_std=noise.std(unbiased=False),target_noise_max=torch.abs(noise).max())

__all__=["DeterministicUpdateMetrics","ParameterSharingMADDPG","ParameterSharingMATD3"]
