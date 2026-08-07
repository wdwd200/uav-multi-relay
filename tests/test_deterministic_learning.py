import copy
import numpy as np
import pytest
import torch

from uav_multi_relay.learning import (CentralizedCritic, CentralizedTwinCritic, ParameterSharingMADDPG, ParameterSharingMATD3, ReplayBatch, SharedDeterministicActor)

def _batch(terminated=0.0,truncated=0.0):
    torch.manual_seed(4); n,k,l,g,a=5,4,6,9,3
    return ReplayBatch(torch.randn(n,k,l),torch.randn(n,g),torch.full((n,k,a),.25),torch.ones(n,1),torch.randn(n,k,l),torch.randn(n,g),torch.full((n,1),terminated),torch.full((n,1),truncated))
def _agent(cls): return cls(6,9,4,hidden_dims=(16,16))
def test_deterministic_network_shapes_ranges_and_repeatability():
    x=torch.randn(3,4,6); actor=SharedDeterministicActor(6,3,(8,)); out=actor(x)
    assert out.shape==(3,4,3) and torch.all(out<=1) and torch.all(out>=-1) and torch.equal(out,actor(x))
    assert CentralizedCritic(9,4,3,(8,))(torch.randn(3,9),out).shape==(3,1)
    assert all(q.shape==(3,1) for q in CentralizedTwinCritic(9,4,3,(8,))(torch.randn(3,9),out))
def test_maddpg_termination_mask_and_update_changes_networks():
    agent=_agent(ParameterSharingMADDPG); batch=_batch(); target=agent.compute_target_q(batch); terminated=agent.compute_target_q(_batch(1.0)); assert torch.allclose(terminated,torch.ones_like(terminated)); assert not torch.allclose(target,terminated)
    before=copy.deepcopy(agent.actor.state_dict()); metrics=agent.update(batch); assert metrics.actor_updated==1 and metrics.td_error_mean>=0 and any(not torch.equal(before[k],agent.actor.state_dict()[k]) for k in before)
def test_matd3_delay_noise_and_applied_action_semantics():
    agent=_agent(ParameterSharingMATD3); batch=_batch(); before=copy.deepcopy(agent.actor.state_dict()); first=agent.update(batch); assert first.actor_updated==0 and all(torch.equal(before[k],agent.actor.state_dict()[k]) for k in before)
    second=agent.update(batch); assert second.actor_updated==1 and second.target_noise_max<=agent.noise_clip+1e-7 and all(np.isfinite(getattr(second,n)) for n in second.__dataclass_fields__)
    replacement=ReplayBatch(batch.local_observations,batch.global_states,torch.zeros_like(batch.applied_actions),batch.rewards,batch.next_local_observations,batch.next_global_states,batch.terminated,batch.truncated)
    assert not torch.allclose(agent.critic(batch.global_states,batch.applied_actions)[0],agent.critic(replacement.global_states,replacement.applied_actions)[0])
def test_dynamic_single_relay_supported():
    assert _agent(ParameterSharingMADDPG).act(np.zeros((4,6),dtype=np.float32)).shape==(4,3)
    assert ParameterSharingMATD3(6,9,1,hidden_dims=(8,)).act(np.zeros((1,6),dtype=np.float32)).shape==(1,3)
