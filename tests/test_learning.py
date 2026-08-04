import subprocess
import sys
import numpy as np
import pytest
import torch

from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.learning import (
    CentralizedTwinCritic,
    MultiAgentReplayBuffer,
    ReplayBatch,
    SharedGaussianActor,
)


def test_default_environment_dimensions_feed_learning_networks() -> None:
    env = MultiRelayEnvironment()
    observation, _ = env.reset(seed=0)
    num_relays = observation["local"].shape[0]
    local_observation_dim = observation["local"].shape[-1]
    global_state_dim = observation["global"].shape[-1]
    actor = SharedGaussianActor(local_observation_dim=local_observation_dim)
    critic = CentralizedTwinCritic(
        global_state_dim=global_state_dim,
        num_relays=num_relays,
    )
    assert num_relays == 4
    assert local_observation_dim == 23
    assert global_state_dim == 42
    mean, log_std = actor(torch.as_tensor(observation["local"])[None])
    assert mean.shape == log_std.shape == (1, num_relays, 3)
    q1, q2 = critic(
        torch.as_tensor(observation["global"])[None],
        torch.zeros(1, num_relays, 3),
    )
    assert q1.shape == q2.shape == (1, 1)


def test_actor_shapes_finiteness_and_bounds() -> None:
    actor = SharedGaussianActor()
    observations = torch.randn(5, 4, 23)
    actions, log_probability = actor.sample(observations)
    assert actions.shape == (5, 4, 3)
    assert log_probability.shape == (5, 4, 1)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_probability).all()
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)


def test_actor_backbone_has_exact_hidden_linear_layers() -> None:
    default_actor = SharedGaussianActor()
    custom_actor = SharedGaussianActor(hidden_dims=(64, 32, 16))
    default_count = sum(isinstance(module, torch.nn.Linear) for module in default_actor.backbone)
    custom_count = sum(isinstance(module, torch.nn.Linear) for module in custom_actor.backbone)
    assert default_count == 2
    assert custom_count == 3


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
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in actor.parameters()
    )


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
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in critic.q1_net.parameters()
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in critic.q2_net.parameters()
    )
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


@pytest.mark.parametrize("num_relays", [1, 8])
def test_networks_support_dynamic_relay_counts(num_relays: int) -> None:
    env = MultiRelayEnvironment()
    env.config = env.config.__class__(
        num_relays,
        env.config.delta_t_s,
        env.config.max_steps,
        env.config.relay_motion_limits,
        env.config.high_motion_limits,
        env.config.low_motion_limits,
        env.config.flight_bounds,
        env.config.hard_safety_distance_m,
        env.config.soft_safety_distance_m,
        env.config.hard_max_link_distance_m,
        env.config.rate_reference_bps,
        env.config.channel,
        env.config.high_trajectory,
        env.config.low_trajectory,
    )
    observation, _ = env.reset(seed=0)
    local_dim = observation["local"].shape[-1]
    global_dim = observation["global"].shape[-1]
    actor = SharedGaussianActor(local_observation_dim=local_dim)
    critic = CentralizedTwinCritic(global_dim, num_relays)
    actions, log_probability = actor.sample(torch.randn(2, num_relays, local_dim))
    q1, q2 = critic(torch.randn(2, global_dim), actions)
    assert actions.shape == (2, num_relays, 3)
    assert log_probability.shape == (2, num_relays, 1)
    assert q1.shape == q2.shape == (2, 1)


def test_base_package_import_does_not_import_learning_or_torch() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import uav_multi_relay; print('torch' in sys.modules)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


def _replay_buffer(capacity: int = 4, seed: int | None = 0) -> MultiAgentReplayBuffer:
    return MultiAgentReplayBuffer(
        capacity=capacity,
        num_relays=2,
        local_observation_dim=3,
        global_state_dim=4,
        action_dim=3,
        seed=seed,
    )


def _add_transition(buffer: MultiAgentReplayBuffer, value: float, **flags: object) -> None:
    buffer.add(
        np.full((2, 3), value, dtype=np.float32),
        np.full(4, value, dtype=np.float32),
        np.full((2, 3), value / 10.0, dtype=np.float32),
        value,
        np.full((2, 3), value + 1.0, dtype=np.float32),
        np.full(4, value + 1.0, dtype=np.float32),
        flags.get("terminated", False),
        flags.get("truncated", False),
    )


def test_replay_buffer_stores_copied_transitions_and_samples_float32_batches() -> None:
    buffer = _replay_buffer()
    assert len(buffer) == 0
    local = np.ones((2, 3), dtype=np.float32)
    buffer.add(local, np.ones(4), np.full((2, 3), 0.5), 1.0, local + 1, np.full(4, 2.0), False, True)
    local[:] = 99.0
    assert len(buffer) == 1
    batch = buffer.sample(1)
    assert isinstance(batch, ReplayBatch)
    assert batch.local_observations.shape == (1, 2, 3)
    assert batch.global_states.shape == (1, 4)
    assert batch.applied_actions.shape == (1, 2, 3)
    assert batch.rewards.shape == batch.terminated.shape == batch.truncated.shape == (1, 1)
    for tensor in vars(batch).values():
        assert tensor.dtype == torch.float32
        assert not tensor.requires_grad
        assert torch.isfinite(tensor).all()
    assert torch.all(batch.applied_actions >= -1.0) and torch.all(batch.applied_actions <= 1.0)
    assert torch.all(batch.local_observations == 1.0)
    assert batch.terminated.item() == 0.0
    assert batch.truncated.item() == 1.0


def test_replay_buffer_overwrites_circularly_and_samples_deterministically() -> None:
    first = _replay_buffer(capacity=2, seed=12)
    second = _replay_buffer(capacity=2, seed=12)
    for value in range(3):
        _add_transition(first, float(value))
        _add_transition(second, float(value))
    assert len(first) == 2
    assert first.position == 1
    assert set(first.global_states[:, 0]) == {1.0, 2.0}
    first_batch = first.sample(2)
    second_batch = second.sample(2)
    assert torch.equal(first_batch.global_states, second_batch.global_states)
    assert torch.unique(first_batch.global_states[:, 0]).numel() == 2


def test_replay_buffer_rejects_invalid_input_and_insufficient_samples() -> None:
    buffer = _replay_buffer()
    with pytest.raises(ValueError):
        buffer.sample(1)
    with pytest.raises(ValueError):
        MultiAgentReplayBuffer(0, 2, 3, 4, 3)
    with pytest.raises(ValueError):
        buffer.add(np.zeros((2, 3)), np.zeros(4), np.full((2, 3), 1.1), 0.0, np.zeros((2, 3)), np.zeros(4), False, False)
    with pytest.raises(ValueError):
        buffer.add(np.zeros((2, 2)), np.zeros(4), np.zeros((2, 3)), 0.0, np.zeros((2, 3)), np.zeros(4), False, False)


def test_environment_applied_actions_create_a_real_replay_transition() -> None:
    env = MultiRelayEnvironment()
    observation, _ = env.reset(seed=0)
    buffer = MultiAgentReplayBuffer(
        capacity=2,
        num_relays=observation["local"].shape[0],
        local_observation_dim=observation["local"].shape[-1],
        global_state_dim=observation["global"].shape[-1],
        action_dim=3,
        seed=0,
    )
    next_observation, reward, terminated, truncated, info = env.step(np.ones((4, 3)))
    buffer.add(
        observation["local"],
        observation["global"],
        info["applied_relay_actions"],
        reward,
        next_observation["local"],
        next_observation["global"],
        terminated,
        truncated,
    )
    batch = buffer.sample(1)
    assert torch.allclose(batch.applied_actions[0], torch.as_tensor(info["applied_relay_actions"], dtype=torch.float32))
    assert batch.terminated.item() == float(terminated)
    assert batch.truncated.item() == float(truncated)
