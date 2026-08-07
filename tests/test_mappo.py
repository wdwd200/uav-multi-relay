import copy
import numpy as np
import pytest
import torch

from uav_multi_relay.learning import CentralizedValueCritic, MAPPOAgent, MAPPOConfig, MAPPORollout, SharedGaussianActor, compute_gae

def _rollout(capacity: int = 4, applied_shift: float = 0.0) -> MAPPORollout:
    rollout=MAPPORollout(capacity,2,4,5)
    for index in range(capacity):
        requested=np.full((2,3),0.1*(index+1),dtype=np.float32)
        applied=np.clip(requested+applied_shift,-1,1)
        rollout.add(np.full((2,4),index,dtype=np.float32),np.full(5,index,dtype=np.float32),requested,applied,-1.0,1.0,float(index),float(index+1),False,False)
    return rollout

def test_actor_evaluate_actions_matches_sample_log_probability_and_handles_bounds() -> None:
    torch.manual_seed(3); actor=SharedGaussianActor(4,3,(8,8)); observations=torch.randn(3,2,4)
    actions, per=actor.sample(observations,deterministic=True)
    joint, evaluated, entropy=actor.evaluate_actions(observations,actions)
    assert joint.shape == entropy.shape == (3,1) and evaluated.shape == per.shape == (3,2,1)
    assert torch.allclose(joint,per.sum(dim=1),atol=1e-5) and torch.allclose(evaluated,per,atol=1e-5)
    boundary_joint,_,_=actor.evaluate_actions(observations,torch.ones_like(actions))
    assert torch.isfinite(boundary_joint).all()

def test_value_critic_shape_and_finiteness() -> None:
    critic=CentralizedValueCritic(5,(8,8)); value=critic(torch.randn(4,5))
    assert value.shape == (4,1) and torch.isfinite(value).all()

def test_gae_exact_terminated_and_truncated_semantics() -> None:
    rewards=np.array([1.,2.]); values=np.array([.5,.5]); next_values=np.array([.5,.7])
    advantages,returns=compute_gae(rewards,values,next_values,np.array([0.,1.]),np.zeros(2),.9,.8)
    assert np.allclose(advantages,[2.03,1.5]) and np.allclose(returns,[2.53,2.])
    advantages,returns=compute_gae(rewards,values,next_values,np.zeros(2),np.array([1.,0.]),.9,.8)
    assert np.allclose(advantages,[.95,2.13]) and np.allclose(returns,[1.45,2.63])

def test_gae_multiple_episode_rollout_does_not_cross_boundaries() -> None:
    advantages,_=compute_gae(np.ones(4),np.zeros(4),np.zeros(4),np.array([0.,1.,0.,0.]),np.array([0.,0.,1.,0.]),1.,1.)
    assert np.allclose(advantages,[2.,1.,1.,1.])

def test_rollout_requires_full_capacity_and_exposes_gae_arrays() -> None:
    rollout=_rollout(2)
    arrays=rollout.arrays(.99,.95)
    assert arrays["requested_actions"].shape == (2,2,3) and arrays["advantages"].shape == (2,1)
    rollout.clear()
    with pytest.raises(ValueError): rollout.arrays(.99,.95)

def test_ppo_updates_parameters_with_finite_metrics() -> None:
    agent=MAPPOAgent(4,5,2,hidden_dims=(8,8),config=MAPPOConfig(update_epochs=2,mini_batch_size=2))
    before=[item.detach().clone() for item in agent.actor.parameters()]; metrics=agent.update(_rollout())
    assert all(np.isfinite(value) for value in vars(metrics).values())
    assert any(not torch.equal(left,right) for left,right in zip(before,agent.actor.parameters()))
    assert 0 <= metrics.clip_fraction <= 1 and metrics.actor_gradient_norm >= 0 and metrics.critic_gradient_norm >= 0

def test_applied_actions_do_not_change_ppo_ratio_or_parameters() -> None:
    torch.manual_seed(7); first=MAPPOAgent(4,5,2,hidden_dims=(8,8),config=MAPPOConfig(update_epochs=1,mini_batch_size=4)); second=copy.deepcopy(first)
    torch.manual_seed(8); first.update(_rollout(applied_shift=0.0)); torch.manual_seed(8); second.update(_rollout(applied_shift=.7))
    for one,two in ((first.actor,second.actor),(first.value_critic,second.value_critic)):
        assert all(torch.equal(a,b) for a,b in zip(one.parameters(),two.parameters()))
