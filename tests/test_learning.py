import subprocess
import sys
import numpy as np
import pytest
import torch
import copy

from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.learning import (
    CentralizedTwinCritic,
    MASACUpdateMetrics,
    MultiAgentReplayBuffer,
    ParameterSharingMASAC,
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


def _masac_batch(
    *,
    batch_size: int = 3,
    num_relays: int = 2,
    local_dim: int = 4,
    global_dim: int = 5,
    terminated: object = 0.0,
    truncated: object = 0.0,
) -> ReplayBatch:
    terminated_tensor = torch.as_tensor(terminated, dtype=torch.float32)
    truncated_tensor = torch.as_tensor(truncated, dtype=torch.float32)
    if terminated_tensor.numel() == 1:
        terminated_tensor = terminated_tensor.expand(batch_size, 1).clone()
    if truncated_tensor.numel() == 1:
        truncated_tensor = truncated_tensor.expand(batch_size, 1).clone()
    return ReplayBatch(
        local_observations=torch.randn(batch_size, num_relays, local_dim),
        global_states=torch.randn(batch_size, global_dim),
        applied_actions=torch.tanh(torch.randn(batch_size, num_relays, 3)),
        rewards=torch.randn(batch_size, 1),
        next_local_observations=torch.randn(batch_size, num_relays, local_dim),
        next_global_states=torch.randn(batch_size, global_dim),
        terminated=terminated_tensor,
        truncated=truncated_tensor,
    )


def _small_masac(num_relays: int = 2) -> ParameterSharingMASAC:
    return ParameterSharingMASAC(
        local_observation_dim=4,
        global_state_dim=5,
        num_relays=num_relays,
        hidden_dims=(16, 16),
        device="cpu",
    )


def test_masac_act_is_bounded_float32_and_deterministic_when_requested() -> None:
    masac = _small_masac(num_relays=2)
    observations = np.zeros((2, 4), dtype=np.float32)
    first = masac.act(observations, deterministic=True)
    second = masac.act(observations, deterministic=True)
    assert first.shape == (2, 3)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    assert np.all(first >= -1.0) and np.all(first <= 1.0)
    assert np.array_equal(first, second)
    with pytest.raises(ValueError):
        masac.act(np.zeros((4, 4), dtype=np.float32))


def test_masac_target_critic_is_independent_frozen_and_uses_polyak_updates() -> None:
    masac = _small_masac()
    online_parameters = list(masac.critic.parameters())
    target_parameters = list(masac.target_critic.parameters())
    assert all(not parameter.requires_grad for parameter in target_parameters)
    assert {id(parameter) for parameter in online_parameters}.isdisjoint(
        id(parameter) for parameter in target_parameters
    )
    assert all(torch.equal(online, target) for online, target in zip(online_parameters, target_parameters))
    batch = _masac_batch()
    before_target = [parameter.detach().clone() for parameter in target_parameters]
    masac.update(batch)
    assert any(not torch.equal(before, after) for before, after in zip(before_target, target_parameters))


def test_masac_joint_log_probability_sums_relay_terms() -> None:
    values = torch.tensor([[[1.0], [2.0]], [[-1.0], [0.5]]])
    result = ParameterSharingMASAC._joint_log_probability(values)
    assert result.shape == (2, 1)
    assert torch.equal(result, torch.tensor([[3.0], [-0.5]]))


def test_masac_critic_target_masks_terminated_but_bootstraps_truncated() -> None:
    masac = _small_masac()
    for parameter in masac.target_critic.parameters():
        parameter.data.zero_()
    batch = _masac_batch(
        batch_size=2,
        terminated=torch.tensor([[1.0], [0.0]]),
        truncated=torch.tensor([[0.0], [1.0]]),
    )
    target = masac.compute_critic_target(batch)
    assert target.shape == (2, 1)
    assert torch.isfinite(target).all()
    assert target[0].item() == pytest.approx(batch.rewards[0].item())
    assert target[1].item() != pytest.approx(batch.rewards[1].item())


@pytest.mark.parametrize("num_relays", [1, 8])
def test_masac_update_supports_dynamic_relay_counts_and_returns_finite_metrics(
    num_relays: int,
) -> None:
    masac = _small_masac(num_relays=num_relays)
    batch = _masac_batch(num_relays=num_relays)
    actor_before = [parameter.detach().clone() for parameter in masac.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in masac.critic.parameters()]
    metrics = masac.update(batch)
    assert isinstance(metrics, MASACUpdateMetrics)
    assert all(np.isfinite(value) for value in vars(metrics).values())
    assert metrics.td_error_mean >= 0.0 and metrics.td_error_p95 >= 0.0 and metrics.td_error_max >= 0.0
    assert metrics.actor_gradient_norm >= 0.0 and metrics.critic_gradient_norm >= 0.0
    assert 0.0 <= metrics.actor_action_saturation_rate <= 1.0
    assert metrics.actor_log_std_min <= metrics.actor_log_std_mean <= metrics.actor_log_std_max
    assert masac.alpha.item() > 0.0 and np.isfinite(masac.alpha.item())
    assert any(not torch.equal(before, after) for before, after in zip(actor_before, masac.actor.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(critic_before, masac.critic.parameters()))
    assert all(parameter.grad is None for parameter in masac.critic.parameters())


def test_update_diagnostics_do_not_change_rng_or_optimization_result() -> None:
    torch.manual_seed(123)
    first = _small_masac()
    second = copy.deepcopy(first)
    batch = _masac_batch()
    torch.manual_seed(456)
    first.update(batch)
    torch.manual_seed(456)
    second.update(batch)
    for first_module, second_module in ((first.actor, second.actor), (first.critic, second.critic), (first.target_critic, second.target_critic)):
        assert all(torch.equal(left, right) for left, right in zip(first_module.parameters(), second_module.parameters()))
    assert torch.equal(first.log_alpha, second.log_alpha)
    first_state = first.actor_optimizer.state_dict()
    second_state = second.actor_optimizer.state_dict()
    assert first_state["param_groups"] == second_state["param_groups"]
    assert first_state["state"].keys() == second_state["state"].keys()
    for key in first_state["state"]:
        for name, value in first_state["state"][key].items():
            other = second_state["state"][key][name]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other


def test_masac_rejects_incompatible_batch_shapes() -> None:
    masac = _small_masac()
    batch = _masac_batch()
    invalid = ReplayBatch(
        local_observations=batch.local_observations[:, :1],
        global_states=batch.global_states,
        applied_actions=batch.applied_actions,
        rewards=batch.rewards,
        next_local_observations=batch.next_local_observations,
        next_global_states=batch.next_global_states,
        terminated=batch.terminated,
        truncated=batch.truncated,
    )
    with pytest.raises(ValueError):
        masac.update(invalid)
