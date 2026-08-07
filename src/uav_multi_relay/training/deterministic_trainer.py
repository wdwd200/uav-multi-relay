"""Shared replay-based collection loop for MATD3 and MADDPG."""
from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
from typing import Callable
import numpy as np
from ..environment import MultiRelayEnvironment
from ..learning import DeterministicUpdateMetrics, MultiAgentReplayBuffer, ParameterSharingMADDPG, ParameterSharingMATD3
from .trainer import _observation_arrays

DeterministicAgent = ParameterSharingMADDPG | ParameterSharingMATD3

@dataclass(frozen=True)
class DeterministicTrainingConfig:
    total_environment_steps: int
    replay_capacity: int = 100_000
    batch_size: int = 256
    random_action_steps: int = 2_000
    update_after_steps: int = 2_000
    updates_per_step: int = 1
    exploration_noise_std: float = 0.1
    seed: int = 0
    def __post_init__(self)->None:
        for name in ("total_environment_steps","replay_capacity","batch_size","updates_per_step"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,Integral) or value<=0: raise ValueError(f"{name} must be a positive integer")
        for name in ("random_action_steps","update_after_steps"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,Integral) or value<0: raise ValueError(f"{name} must be a non-negative integer")
        if not np.isfinite(self.exploration_noise_std) or self.exploration_noise_std<0: raise ValueError("exploration_noise_std must be finite and non-negative")
        if self.batch_size>self.replay_capacity: raise ValueError("batch_size must not exceed replay_capacity")

@dataclass(frozen=True)
class DeterministicTrainingProgress:
    environment_steps:int; total_updates:int; completed_episodes:int; replay_size:int; mean_rate_e2e_bps:float; termination_rate:float; intervention_rate:float; requested_applied_mismatch_rate:float; last_update_metrics:DeterministicUpdateMetrics|None
@dataclass(frozen=True)
class DeterministicTrainingSummary:
    total_environment_steps:int; total_updates:int; completed_episodes:int; episode_returns:tuple[float,...]; episode_lengths:tuple[int,...]; mean_rate_e2e_bps:float; termination_rate:float; intervention_rate:float; requested_applied_mismatch_rate:float; last_update_metrics:DeterministicUpdateMetrics|None

def train_deterministic(env:MultiRelayEnvironment,agent:DeterministicAgent,replay_buffer:MultiAgentReplayBuffer,config:DeterministicTrainingConfig,*,progress_interval_steps:int|None=None,progress_callback:Callable[[DeterministicTrainingProgress],None]|None=None)->DeterministicTrainingSummary:
    if not isinstance(agent,(ParameterSharingMADDPG,ParameterSharingMATD3)) or not isinstance(config,DeterministicTrainingConfig): raise ValueError("incompatible deterministic training inputs")
    if replay_buffer.capacity!=config.replay_capacity: raise ValueError("replay_buffer.capacity must equal config.replay_capacity")
    if progress_callback is not None and (not callable(progress_callback) or not isinstance(progress_interval_steps,Integral) or progress_interval_steps<=0): raise ValueError("progress callback requires positive interval")
    observation,_=env.reset(seed=config.seed); local,global_state=_observation_arrays(observation)
    if local.shape!=(agent.num_relays,agent.local_observation_dim) or global_state.shape!=(agent.global_state_dim,): raise ValueError("agent and environment observations are incompatible")
    rng=np.random.default_rng(config.seed); returns:list[float]=[]; lengths:list[int]=[]; rates:list[float]=[]; current_return=0.; current_length=0; completed=updates=terminations=interventions=mismatches=0; last=None
    for step in range(config.total_environment_steps):
        if step<config.random_action_steps: requested=rng.uniform(-1,1,(agent.num_relays,agent.action_dim)).astype(np.float32)
        else: requested=np.clip(agent.act(local)+rng.normal(0,config.exploration_noise_std,(agent.num_relays,agent.action_dim)), -1,1).astype(np.float32)
        next_observation,reward,terminated,truncated,info=env.step(requested); next_local,next_global=_observation_arrays(next_observation); applied=np.asarray(info["applied_relay_actions"],dtype=np.float32); replay_buffer.add(local,global_state,applied,reward,next_local,next_global,terminated,truncated)
        rate=float(info["rate_e2e_bps"]); norms=np.asarray(info["intervention_norms"],dtype=float); mismatch=np.linalg.norm(requested-applied,axis=1)
        if not np.isfinite(reward) or not np.isfinite(rate) or not np.all(np.isfinite(norms)) or not np.all(np.isfinite(mismatch)): raise ValueError("environment returned non-finite training values")
        rates.append(rate); current_return+=float(reward); current_length+=1; interventions+=int(np.any(norms>1e-9)); mismatches+=int(np.any(mismatch>1e-6)); terminations+=int(terminated)
        if step+1>=config.update_after_steps and replay_buffer.size>=config.batch_size:
            for _ in range(config.updates_per_step): last=agent.update(replay_buffer.sample(config.batch_size,device=agent.device)); updates+=1
        local,global_state=next_local,next_global
        if terminated or truncated:
            returns.append(current_return); lengths.append(current_length); completed+=1; current_return=0.; current_length=0; observation,_=env.reset(seed=config.seed+completed); local,global_state=_observation_arrays(observation)
        steps=step+1
        if progress_callback is not None and (steps%int(progress_interval_steps)==0 or steps==config.total_environment_steps):
            progress_callback(DeterministicTrainingProgress(steps,updates,completed,replay_buffer.size,float(np.mean(rates)),float(terminations/steps),float(interventions/steps),float(mismatches/steps),last))
    values=(float(np.mean(rates)),terminations/config.total_environment_steps,interventions/config.total_environment_steps,mismatches/config.total_environment_steps,*returns)
    if not all(np.isfinite(x) for x in values): raise ValueError("training summary contains non-finite values")
    return DeterministicTrainingSummary(config.total_environment_steps,updates,completed,tuple(returns),tuple(lengths),float(np.mean(rates)),float(terminations/config.total_environment_steps),float(interventions/config.total_environment_steps),float(mismatches/config.total_environment_steps),last)

__all__=["DeterministicTrainingConfig","DeterministicTrainingProgress","DeterministicTrainingSummary","train_deterministic"]
