import copy
import math

import numpy as np
import pytest
import torch

from uav_multi_relay.learning import (
    CentralizedValueCritic,
    MAPPOAgent,
    MAPPOConfig,
    MAPPORollout,
    SharedGaussianActor,
    compute_gae,
    per_relay_ratio,
)


def _rollout(capacity: int = 4, applied_shift: float = 0.0) -> MAPPORollout:
    rollout = MAPPORollout(capacity, 2, 4, 5)
    for index in range(capacity):
        requested = np.full((2, 3), 0.1 * (index + 1), dtype=np.float32)
        applied = np.clip(requested + applied_shift, -1.0, 1.0)
        old_log_probability = np.full((2, 1), -1.0, dtype=np.float32)
        rollout.add(
            np.full((2, 4), index, dtype=np.float32),
            np.full(5, index, dtype=np.float32),
            requested,
            applied,
            old_log_probability,
            1.0,
            float(index),
            float(index + 1),
            False,
            False,
        )
    return rollout


def test_actor_sample_and_evaluate_match_per_relay_for_random_actions() -> None:
    torch.manual_seed(3)
    actor = SharedGaussianActor(4, 3, (8, 8))
    observations = torch.randn(3, 2, 4)
    actions, sampled = actor.sample(observations)
    evaluated, entropy, joint = actor.evaluate_actions(observations, actions)
    assert sampled.shape == evaluated.shape == entropy.shape == (3, 2, 1)
    assert joint.shape == (3, 1)
    assert torch.allclose(sampled, evaluated, atol=1e-5)
    assert torch.allclose(joint, evaluated.sum(dim=1), atol=1e-6)


def test_actor_entropy_is_finite_distribution_property_and_handles_bounds() -> None:
    actor = SharedGaussianActor(4, 3, (8, 8))
    observations = torch.randn(2, 2, 4)
    actions, _ = actor.sample(observations, deterministic=True)
    _, entropy, _ = actor.evaluate_actions(observations, actions)
    _, boundary_entropy, _ = actor.evaluate_actions(observations, torch.ones_like(actions))
    assert entropy.shape == boundary_entropy.shape == (2, 2, 1)
    assert torch.isfinite(entropy).all() and torch.isfinite(boundary_entropy).all()
    with torch.no_grad():
        actor.log_std_head.bias.add_(1.0)
    _, higher_entropy, _ = actor.evaluate_actions(observations, actions)
    assert higher_entropy.mean() > entropy.mean()
    changed_actions = -actions
    _, same_distribution_entropy, _ = actor.evaluate_actions(observations, changed_actions)
    assert torch.allclose(higher_entropy, same_distribution_entropy)


def test_value_critic_shape_and_finiteness() -> None:
    critic = CentralizedValueCritic(5, (8, 8))
    value = critic(torch.randn(4, 5))
    assert value.shape == (4, 1)
    assert torch.isfinite(value).all()


def test_gae_exact_terminated_and_truncated_semantics() -> None:
    rewards = np.array([1.0, 2.0])
    values = np.array([0.5, 0.5])
    next_values = np.array([0.5, 0.7])
    advantages, returns = compute_gae(rewards, values, next_values, np.array([0.0, 1.0]), np.zeros(2), 0.9, 0.8)
    assert np.allclose(advantages, [2.03, 1.5])
    assert np.allclose(returns, [2.53, 2.0])
    advantages, returns = compute_gae(rewards, values, next_values, np.zeros(2), np.array([1.0, 0.0]), 0.9, 0.8)
    assert np.allclose(advantages, [0.95, 2.13])
    assert np.allclose(returns, [1.45, 2.63])


def test_gae_multiple_episode_rollout_does_not_cross_boundaries() -> None:
    advantages, _ = compute_gae(np.ones(4), np.zeros(4), np.zeros(4), np.array([0.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0]), 1.0, 1.0)
    assert np.allclose(advantages, [2.0, 1.0, 1.0, 1.0])


def test_rollout_preserves_per_relay_log_probabilities() -> None:
    arrays = _rollout(2).arrays(0.99, 0.95)
    assert arrays["old_per_relay_log_probabilities"].shape == (2, 2, 1)
    assert arrays["requested_actions"].shape == (2, 2, 3)


def test_per_relay_ratio_does_not_collapse_to_joint_ratio() -> None:
    old = torch.zeros(1, 2, 1)
    new = torch.tensor([[[math.log(1.1)], [math.log(0.9)]]])
    ratio = per_relay_ratio(new, old)
    assert torch.allclose(ratio, torch.tensor([[[1.1], [0.9]]]), atol=1e-6)
    assert not torch.allclose(ratio, torch.full_like(ratio, 0.99))


def test_single_relay_per_relay_ratio_matches_standard_ppo_ratio() -> None:
    old = torch.tensor([[[0.2]], [[-0.4]]])
    new = torch.tensor([[[0.3]], [[-0.1]]])
    assert torch.allclose(per_relay_ratio(new, old).squeeze(1), torch.exp(new.squeeze(1) - old.squeeze(1)))


def test_ppo_updates_parameters_with_finite_metrics() -> None:
    agent = MAPPOAgent(4, 5, 2, hidden_dims=(8, 8), config=MAPPOConfig(update_epochs=2, mini_batch_size=2))
    before = [item.detach().clone() for item in agent.actor.parameters()]
    metrics = agent.update(_rollout())
    assert all(np.isfinite(value) for value in vars(metrics).values())
    assert any(not torch.equal(left, right) for left, right in zip(before, agent.actor.parameters()))
    assert 0.0 <= metrics.clip_fraction <= 1.0


def test_applied_actions_only_change_mismatch_diagnostics() -> None:
    torch.manual_seed(7)
    first = MAPPOAgent(4, 5, 2, hidden_dims=(8, 8), config=MAPPOConfig(update_epochs=1, mini_batch_size=4))
    second = copy.deepcopy(first)
    torch.manual_seed(8)
    first_metrics = first.update(_rollout(applied_shift=0.0))
    torch.manual_seed(8)
    second_metrics = second.update(_rollout(applied_shift=0.7))
    for one, two in ((first.actor, second.actor), (first.value_critic, second.value_critic)):
        assert all(torch.equal(left, right) for left, right in zip(one.parameters(), two.parameters()))
    for name in ("policy_loss", "approx_kl", "clip_fraction"):
        assert getattr(first_metrics, name) == pytest.approx(getattr(second_metrics, name))
    assert first_metrics.requested_applied_mismatch_mean != second_metrics.requested_applied_mismatch_mean


def test_relay_permutation_preserves_actor_update_metrics() -> None:
    torch.manual_seed(15)
    first = MAPPOAgent(4, 5, 2, hidden_dims=(8, 8), config=MAPPOConfig(update_epochs=1, mini_batch_size=4))
    second = copy.deepcopy(first)
    original = _rollout()
    permuted = MAPPORollout(4, 2, 4, 5)
    arrays = original.arrays(0.99, 0.95)
    for index in range(4):
        permuted.add(arrays["local_observations"][index, ::-1].copy(), arrays["global_states"][index], arrays["requested_actions"][index, ::-1].copy(), arrays["applied_actions"][index, ::-1].copy(), arrays["old_per_relay_log_probabilities"][index, ::-1].copy(), arrays["rewards"][index, 0], arrays["values"][index, 0], arrays["next_values"][index, 0], arrays["terminated"][index, 0], arrays["truncated"][index, 0])
    torch.manual_seed(16)
    first_metrics = first.update(original)
    torch.manual_seed(16)
    second_metrics = second.update(permuted)
    assert first_metrics.policy_loss == pytest.approx(second_metrics.policy_loss)
    assert first_metrics.entropy == pytest.approx(second_metrics.entropy)
