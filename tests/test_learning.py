import numpy as np
import pytest
import torch

from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.learning import CentralizedTwinCritic, SharedGaussianActor


def test_default_environment_dimensions_feed_learning_networks() -> None:
    env = MultiRelayEnvironment()
    observation, _ = env.reset(seed=0)
    assert observation["local"].shape == (4, 23)
    assert observation["global"].shape == (42,)


def test_actor_shapes_finiteness_and_bounds() -> None:
    actor = SharedGaussianActor()
    observations = torch.randn(5, 4, 23)
    actions, log_probability = actor.sample(observations)
    assert actions.shape == (5, 4, 3)
    assert log_probability.shape == (5, 4, 1)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_probability).all()
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)


def test_actor_deterministic_action_equals_tanh_mean() -> None:
    actor = SharedGaussianActor()
    observations = torch.randn(2, 4, 23)
    mean, _ = actor(observations)
    actions, _ = actor.sample(observations, deterministic=True)
    assert torch.allclose(actions, torch.tanh(mean))


def test_actor_rsample_path_provides_finite_parameter_gradients() -> None:
    actor = SharedGaussianActor()
    observations = torch.randn(3, 4, 23)
    actions, log_probability = actor.sample(observations)
    (actions.square().mean() + log_probability.mean()).backward()
    gradients = [parameter.grad for parameter in actor.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_actor_rejects_invalid_shapes() -> None:
    actor = SharedGaussianActor()
    with pytest.raises(ValueError):
        actor(torch.randn(4, 23))
    with pytest.raises(ValueError):
        actor(torch.randn(5, 4, 22))


def test_critic_shapes_independence_and_finite_gradients() -> None:
    critic = CentralizedTwinCritic(global_state_dim=42, num_relays=4)
    state = torch.randn(5, 42)
    actions = torch.randn(5, 4, 3)
    q1, q2 = critic(state, actions)
    assert q1.shape == (5, 1)
    assert q2.shape == (5, 1)
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()
    q1.sum().backward(retain_graph=True)
    q2.sum().backward()
    assert any(parameter.grad is not None for parameter in critic.q1_net.parameters())
    assert any(parameter.grad is not None for parameter in critic.q2_net.parameters())
    q1_parameters = {id(parameter) for parameter in critic.q1_net.parameters()}
    q2_parameters = {id(parameter) for parameter in critic.q2_net.parameters()}
    assert q1_parameters.isdisjoint(q2_parameters)


def test_critic_rejects_invalid_shapes() -> None:
    critic = CentralizedTwinCritic(global_state_dim=42, num_relays=4)
    with pytest.raises(ValueError):
        critic(torch.randn(5, 41), torch.randn(5, 4, 3))
    with pytest.raises(ValueError):
        critic(torch.randn(5, 42), torch.randn(5, 3))
    with pytest.raises(ValueError):
        critic(torch.randn(5, 42), torch.randn(4, 4, 3))
