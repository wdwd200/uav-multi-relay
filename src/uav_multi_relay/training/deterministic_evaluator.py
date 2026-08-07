"""Side-effect-free deterministic-policy evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
import copy, numpy as np
from ..environment import MultiRelayEnvironment
from ..learning import ParameterSharingMADDPG, ParameterSharingMATD3
from .trainer import _observation_arrays
@dataclass(frozen=True)
class DeterministicEvaluationConfig:
    episodes:int=10; seed:int=0
    def __post_init__(self)->None:
        if isinstance(self.episodes,bool) or not isinstance(self.episodes,Integral) or self.episodes<=0: raise ValueError("episodes must be positive")
@dataclass(frozen=True)
class DeterministicEvaluationSummary:
    mean_return:float; return_std:float; mean_rate_e2e_bps:float; minimum_rate_e2e_bps:float; mean_intervention_rate:float; terminated_episode_rate:float
def evaluate_deterministic(env:MultiRelayEnvironment,agent:ParameterSharingMADDPG|ParameterSharingMATD3,config:DeterministicEvaluationConfig)->DeterministicEvaluationSummary:
    returns=[]; rates=[]; minima=[]; interventions=[]; terminateds=[]
    for index in range(config.episodes):
        current=copy.deepcopy(env); observation,_=current.reset(seed=config.seed+index); local,_=_observation_arrays(observation); ret=0.; rs=[]; iv=0; steps=0; terminated=truncated=False
        while not(terminated or truncated):
            observation,reward,terminated,truncated,info=current.step(agent.act(local)); local,_=_observation_arrays(observation); ret+=float(reward); rs.append(float(info["rate_e2e_bps"])); iv+=int(np.any(np.asarray(info["intervention_norms"],dtype=float)>1e-9)); steps+=1
        returns.append(ret);rates.append(float(np.mean(rs)));minima.append(float(np.min(rs)));interventions.append(iv/steps);terminateds.append(float(terminated))
    values=[np.mean(returns),np.std(returns),np.mean(rates),np.min(minima),np.mean(interventions),np.mean(terminateds)]
    if not all(np.isfinite(x) for x in values): raise ValueError("evaluation results are non-finite")
    return DeterministicEvaluationSummary(*(float(x) for x in values))
__all__=["DeterministicEvaluationConfig","DeterministicEvaluationSummary","evaluate_deterministic"]
