import copy

import numpy as np
import pytest

from uav_multi_relay.baselines import (
    equal_spacing_actions,
    greedy_one_step_actions,
    stationary_actions,
    weighted_spacing_actions,
)
from uav_multi_relay.config import (
    ChannelConfig,
    EndpointTrajectoryConfig,
    EnvironmentConfig,
    FlightBounds,
    default_environment_config,
)
from uav_multi_relay.core import MotionLimits, UAVState
from uav_multi_relay.environment import MultiRelayEnvironment
from uav_multi_relay.safety import (
    NoFeasibleActionError,
    filter_relay_velocities,
    normalized_action_to_velocity,
    velocity_to_normalized_action,
)
from uav_multi_relay.trajectories import WaypointFollower


def test_default_configuration_is_valid_and_frozen() -> None:
    config = default_environment_config()
    assert config.num_relays == 4
    assert config.delta_t_s == pytest.approx(0.2)
    assert config.max_steps == 500
    assert config.flight_bounds.contains([0.0, 0.0, 100.0])
    with pytest.raises(ValueError):
        config.flight_bounds.minimum_m[0] = -1.0
    assert config.channel.reference_gain_linear > 0.0


def test_configuration_rejects_invalid_bounds_and_channel() -> None:
    with pytest.raises(ValueError):
        FlightBounds([0.0, 0.0, 0.0], [1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        ChannelConfig(2.4e9, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0)
    with pytest.raises(ValueError):
        EnvironmentConfig(
            0,
            0.2,
            10,
            MotionLimits(1, 1, 1, 1, 1),
            MotionLimits(1, 1, 1, 1, 1),
            MotionLimits(1, 1, 1, 1, 1),
            FlightBounds([-1, -1, -1], [1, 1, 1]),
            1,
            2,
            5,
            1,
            ChannelConfig(2.4e9, 1, 2, 1, 1, 1, 1, 1, 0.1, 1),
        )


def test_endpoint_trajectory_configuration_is_validated() -> None:
    with pytest.raises(ValueError):
        EndpointTrajectoryConfig(10.0, 10.0, 3.0, 4, 2.0)
    with pytest.raises(ValueError):
        EndpointTrajectoryConfig(10.0, 20.0, 2.0, 4, 2.0)
    with pytest.raises(ValueError):
        EndpointTrajectoryConfig(10.0, 20.0, 3.0, 1, 2.0)
    with pytest.raises(ValueError):
        EnvironmentConfig(
            1,
            0.2,
            10,
            MotionLimits(1, 1, 1, 1, 1),
            MotionLimits(1, 1, 1, 1, 1),
            MotionLimits(1, 1, 1, 1, 1),
            FlightBounds([-10, -10, -10], [10, 10, 10]),
            5,
            5,
            4,
            1,
            ChannelConfig(2.4e9, 1, 2, 1, 1, 1, 1, 1, 0.1, 1),
        )


def test_waypoint_follower_cycles_and_resets() -> None:
    follower = WaypointFollower([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    limits = MotionLimits(5.0, 2.0, 2.0, 10.0, 10.0)
    state = UAVState("H", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    velocity = follower.velocity_for(state, limits, 1.0)
    assert velocity[0] > 0.0
    follower.reset()
    assert follower.velocity_for(state, limits, 1.0)[0] > 0.0


def test_waypoint_arrival_uses_three_dimensional_tolerance() -> None:
    state = UAVState("H", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    follower = WaypointFollower([[1.5, 0.0, 1.5]], arrival_tolerance_m=2.0)
    velocity = follower.velocity_for(
        state, MotionLimits(5.0, 2.0, 2.0, 10.0, 10.0), 1.0
    )
    assert np.linalg.norm(velocity) > 0.0


def test_same_seed_reproduces_initialization_and_endpoint_trajectories() -> None:
    first = MultiRelayEnvironment()
    second = MultiRelayEnvironment()
    first.reset(seed=11)
    second.reset(seed=11)
    for _ in range(20):
        _, _, first_terminated, _, first_info = first.step(np.zeros((4, 3)))
        _, _, second_terminated, _, second_info = second.step(np.zeros((4, 3)))
        assert np.allclose(first_info["positions_m"][[0, -1]], second_info["positions_m"][[0, -1]])
        assert first_terminated == second_terminated == False


def test_different_seeds_produce_different_endpoint_trajectories() -> None:
    first = MultiRelayEnvironment()
    second = MultiRelayEnvironment()
    first.reset(seed=1)
    second.reset(seed=2)
    differences = []
    for _ in range(20):
        _, _, _, _, first_info = first.step(np.zeros((4, 3)))
        _, _, _, _, second_info = second.step(np.zeros((4, 3)))
        differences.append(np.linalg.norm(first_info["positions_m"][[0, -1]] - second_info["positions_m"][[0, -1]]))
    assert any(difference > 1e-9 for difference in differences)


@pytest.mark.parametrize("num_relays", [1, 4, 8])
def test_variable_relay_count_initialization_is_hard_feasible(num_relays: int) -> None:
    config = default_environment_config()
    config = EnvironmentConfig(
        num_relays,
        config.delta_t_s,
        config.max_steps,
        config.relay_motion_limits,
        config.high_motion_limits,
        config.low_motion_limits,
        config.flight_bounds,
        config.hard_safety_distance_m,
        config.soft_safety_distance_m,
        config.hard_max_link_distance_m,
        config.rate_reference_bps,
        config.channel,
    )
    env = MultiRelayEnvironment(config)
    _, info = env.reset(seed=0)
    assert info["positions_m"].shape == (num_relays + 2, 3)
    assert np.all(info["positions_m"] >= config.flight_bounds.minimum_m)
    assert np.all(info["positions_m"] <= config.flight_bounds.maximum_m)
    positions = info["positions_m"]
    pairwise = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    pairwise[np.diag_indices_from(pairwise)] = np.inf
    assert np.min(pairwise) >= config.hard_safety_distance_m
    assert np.all(np.linalg.norm(np.diff(positions, axis=0), axis=1) <= config.hard_max_link_distance_m)


def test_small_custom_bounds_are_supported_when_chain_is_feasible() -> None:
    base = default_environment_config()
    endpoint_limits = MotionLimits(2.0, 1.0, 1.0, 2.0, 2.0)
    config = EnvironmentConfig(
        1,
        base.delta_t_s,
        base.max_steps,
        base.relay_motion_limits,
        endpoint_limits,
        endpoint_limits,
        FlightBounds(np.array([-20.0, -20.0, -20.0]), np.array([20.0, 20.0, 20.0])),
        5.0,
        8.0,
        30.0,
        base.rate_reference_bps,
        base.channel,
        EndpointTrajectoryConfig(5.0, 15.0, 4.0, 4, 2.0),
        EndpointTrajectoryConfig(-15.0, -5.0, 4.0, 4, 2.0),
    )
    env = MultiRelayEnvironment(config)
    _, info = env.reset(seed=4)
    assert np.all(info["positions_m"] >= config.flight_bounds.minimum_m)
    assert np.all(info["positions_m"] <= config.flight_bounds.maximum_m)


def test_impossible_initial_chain_has_clear_value_error() -> None:
    base = default_environment_config()
    config = EnvironmentConfig(
        8,
        base.delta_t_s,
        base.max_steps,
        base.relay_motion_limits,
        base.high_motion_limits,
        base.low_motion_limits,
        FlightBounds(np.zeros(3), np.ones(3)),
        2.0,
        3.0,
        2.0,
        base.rate_reference_bps,
        base.channel,
        EndpointTrajectoryConfig(0.6, 0.9, 0.2, 4, 0.1),
        EndpointTrajectoryConfig(0.1, 0.4, 0.2, 4, 0.1),
    )
    with pytest.raises(ValueError, match="unable to construct"):
        MultiRelayEnvironment(config).reset(seed=0)


def test_default_endpoint_altitudes_are_separated_for_many_seeds() -> None:
    config = default_environment_config()
    for seed in range(100):
        _, info = MultiRelayEnvironment(config).reset(seed=seed)
        high_altitude = info["positions_m"][0, 2]
        low_altitude = info["positions_m"][-1, 2]
        assert config.high_trajectory.altitude_min_m <= high_altitude <= config.high_trajectory.altitude_max_m
        assert config.low_trajectory.altitude_min_m <= low_altitude <= config.low_trajectory.altitude_max_m
        assert high_altitude > low_altitude


@pytest.mark.parametrize("maximum_link_distance", [10.0, 10.001])
def test_constructive_initialization_handles_exact_and_narrow_link_ranges(
    maximum_link_distance: float,
) -> None:
    base = default_environment_config()
    config = EnvironmentConfig(
        4,
        base.delta_t_s,
        base.max_steps,
        base.relay_motion_limits,
        base.high_motion_limits,
        base.low_motion_limits,
        FlightBounds(np.array([-100.0, -100.0, 0.0]), np.array([100.0, 100.0, 100.0])),
        10.0,
        15.0,
        maximum_link_distance,
        base.rate_reference_bps,
        base.channel,
        EndpointTrajectoryConfig(55.0, 60.0, 5.0, 4, 2.0),
        EndpointTrajectoryConfig(40.0, 45.0, 5.0, 4, 2.0),
    )
    for seed in range(20):
        _, info = MultiRelayEnvironment(config).reset(seed=seed)
        links = np.linalg.norm(np.diff(info["positions_m"], axis=0), axis=1)
        assert np.all(links >= 10.0 - 1e-9)
        assert np.all(links <= maximum_link_distance + 1e-9)


@pytest.mark.parametrize("seed", [5169, 6397])
def test_specified_seeds_move_both_endpoints(seed: int) -> None:
    env = MultiRelayEnvironment()
    _, reset_info = env.reset(seed=seed)
    initial_endpoints = reset_info["positions_m"][[0, -1]].copy()
    for _ in range(20):
        _, _, terminated, _, info = env.step(np.zeros((4, 3)))
        assert not terminated
    assert np.linalg.norm(info["positions_m"][[0, -1]] - initial_endpoints) > 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_equal_spacing_baseline_completes_default_episode(seed: int) -> None:
    env = MultiRelayEnvironment()
    env.reset(seed=seed)
    for _ in range(env.config.max_steps):
        _, _, terminated, truncated, info = env.step(equal_spacing_actions(env))
        assert not terminated
        positions = info["positions_m"]
        pairwise = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        pairwise[np.diag_indices_from(pairwise)] = np.inf
        assert np.all(positions >= env.config.flight_bounds.minimum_m)
        assert np.all(positions <= env.config.flight_bounds.maximum_m)
        assert np.min(pairwise) >= env.config.hard_safety_distance_m - 1e-9
        assert np.all(
            np.linalg.norm(np.diff(positions, axis=0), axis=1)
            <= env.config.hard_max_link_distance_m + 1e-9
        )
        assert np.all(np.isfinite(positions))
    assert truncated


def test_reset_is_seed_deterministic_and_observations_have_required_shapes() -> None:
    env = MultiRelayEnvironment()
    first_observation, first_info = env.reset(seed=7)
    second_observation, second_info = env.reset(seed=7)
    assert first_observation["local"].shape == (4, 23)
    assert first_observation["global"].ndim == 1
    assert np.all(np.isfinite(first_observation["global"]))
    assert np.allclose(first_observation["local"], second_observation["local"])
    assert np.allclose(first_info["positions_m"], second_info["positions_m"])


@pytest.mark.parametrize(
    "actions",
    [np.zeros((3, 3)), np.zeros((4, 2)), np.full((4, 3), 1.1), np.full((4, 3), np.nan)],
)
def test_step_rejects_invalid_actions(actions: np.ndarray) -> None:
    env = MultiRelayEnvironment()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(actions)


def test_action_mapping_scales_horizontal_vectors_and_vertical_signs() -> None:
    limits = MotionLimits(10.0, 3.0, 4.0, 1.0, 1.0)
    velocity = normalized_action_to_velocity([1.0, 1.0, -0.5], limits)
    assert np.linalg.norm(velocity[:2]) == pytest.approx(10.0)
    assert velocity[2] == pytest.approx(-2.0)


def test_velocity_to_normalized_action_round_trips_feasible_velocities() -> None:
    limits = MotionLimits(10.0, 3.0, 4.0, 1.0, 1.0)
    velocity = np.array([6.0, -8.0, -2.0])
    normalized = velocity_to_normalized_action(velocity, limits)
    assert np.all(np.isfinite(normalized))
    assert np.all(normalized >= -1.0) and np.all(normalized <= 1.0)
    assert normalized_action_to_velocity(normalized, limits) == pytest.approx(velocity)


@pytest.mark.parametrize(
    "velocity",
    [[10.1, 0.0, 0.0], [0.0, 0.0, 3.1], [0.0, 0.0, -4.1], [np.nan, 0.0, 0.0]],
)
def test_velocity_to_normalized_action_rejects_infeasible_velocities(velocity: list[float]) -> None:
    with pytest.raises(ValueError):
        velocity_to_normalized_action(velocity, MotionLimits(10.0, 3.0, 4.0, 1.0, 1.0))


def test_safety_filter_checks_all_hard_constraints() -> None:
    limits = MotionLimits(10.0, 3.0, 3.0, 20.0, 20.0)
    bounds = FlightBounds([-20.0, -20.0, -20.0], [20.0, 20.0, 20.0])
    relays = (
        UAVState("R1", [-5.0, 0.0, 0.0], np.zeros(3)),
        UAVState("R2", [5.0, 0.0, 0.0], np.zeros(3)),
    )
    result = filter_relay_velocities(
        relays,
        [[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]],
        limits,
        0.1,
        bounds,
        2.0,
        20.0,
        [-15.0, 0.0, 0.0],
        [15.0, 0.0, 0.0],
    )
    candidate_positions = np.stack(
        [relay.position_m + velocity * 0.1 for relay, velocity in zip(relays, result.applied_velocities_mps)]
    )
    assert np.all([bounds.contains(position) for position in candidate_positions])
    assert result.scale <= 1.0
    assert np.all(np.isfinite(result.intervention_norms))


def test_safety_filter_reports_no_feasible_endpoint_chain() -> None:
    relay = (UAVState("R1", [0.0, 0.0, 0.0], np.zeros(3)),)
    with pytest.raises(NoFeasibleActionError):
        filter_relay_velocities(
            relay,
            [[0.0, 0.0, 0.0]],
            MotionLimits(2.0, 2.0, 2.0, 2.0, 2.0),
            0.2,
            FlightBounds([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]),
            5.0,
            20.0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )


def test_step_updates_every_uav_once_and_reports_communication() -> None:
    env = MultiRelayEnvironment()
    observation, reset_info = env.reset(seed=0)
    old_positions = reset_info["positions_m"].copy()
    actions = np.zeros((4, 3))
    observation, reward, terminated, truncated, info = env.step(actions)
    assert observation["local"].shape == (4, 23)
    assert not terminated and not truncated
    assert info["hop_distances_m"].shape == (5,)
    assert info["hop_capacities_bps"].shape == (5,)
    assert info["tdma_fractions"].sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(info["hop_capacities_bps"]))
    assert np.isfinite(reward)
    expected_positions = old_positions + info["velocities_mps"] * env.config.delta_t_s
    assert np.allclose(info["positions_m"], expected_positions)
    assert np.linalg.norm(info["positions_m"][0] - old_positions[0]) > 0.0
    assert np.linalg.norm(info["positions_m"][-1] - old_positions[-1]) > 0.0
    assert np.allclose(
        info["requested_relay_velocities_mps"], actions * env.config.relay_motion_limits.max_horizontal_speed_mps
    )


def test_info_reports_requested_and_executed_normalized_actions_as_copies() -> None:
    env = MultiRelayEnvironment()
    _, reset_info = env.reset(seed=0)
    assert np.array_equal(reset_info["requested_relay_actions"], np.zeros((4, 3)))
    assert np.array_equal(reset_info["applied_relay_actions"], np.zeros((4, 3)))
    requested = np.ones((4, 3))
    _, _, terminated, _, info = env.step(requested)
    assert not terminated
    assert np.array_equal(info["requested_relay_actions"], requested)
    remapped = np.stack(
        [
            velocity_to_normalized_action(velocity, env.config.relay_motion_limits)
            for velocity in info["applied_relay_velocities_mps"]
        ]
    )
    assert np.allclose(info["applied_relay_actions"], remapped)
    assert np.any(info["applied_relay_actions"] != requested)
    applied_velocity = env.relay_states[0].velocity_mps.copy()
    info["applied_relay_actions"][0] = 0.0
    requested[0] = 0.0
    assert np.array_equal(env.relay_states[0].velocity_mps, applied_velocity)
    assert not np.array_equal(info["requested_relay_actions"], requested)


def test_info_arrays_are_copies_and_reward_terms_are_finite() -> None:
    env = MultiRelayEnvironment()
    _, _ = env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros((4, 3)))
    positions = info["positions_m"]
    original = positions.copy()
    positions[0, 0] += 1000.0
    assert np.allclose(env.states[0].position_m, original[0])
    assert np.isfinite(reward)
    assert all(np.isfinite(value) for value in info["reward_terms"].values())


def test_equal_spacing_baseline_returns_bounded_actions() -> None:
    env = MultiRelayEnvironment()
    env.reset(seed=0)
    actions = equal_spacing_actions(env)
    assert actions.shape == (4, 3)
    assert np.all(np.isfinite(actions))
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0)


@pytest.mark.parametrize(
    "baseline",
    [stationary_actions, equal_spacing_actions, weighted_spacing_actions, greedy_one_step_actions],
)
def test_all_baselines_return_valid_actions(baseline: object) -> None:
    env = MultiRelayEnvironment()
    env.reset(seed=0)
    actions = baseline(env)
    assert actions.shape == (4, 3)
    assert np.all(np.isfinite(actions))
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0)


def test_unit_weighted_spacing_matches_equal_spacing() -> None:
    env = MultiRelayEnvironment()
    env.reset(seed=0)
    assert np.allclose(weighted_spacing_actions(env), equal_spacing_actions(env))


@pytest.mark.parametrize("weights", [[1.0, 2.0], [1.0, 0.0, 1.0, 1.0, 1.0], [1.0, np.nan, 1.0, 1.0, 1.0]])
def test_weighted_spacing_rejects_invalid_weights(weights: object) -> None:
    env = MultiRelayEnvironment()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        weighted_spacing_actions(env, weights)


def test_greedy_baseline_does_not_mutate_environment_and_is_not_worse() -> None:
    env = MultiRelayEnvironment()
    env.reset(seed=0)
    stationary_env = copy.deepcopy(env)
    stationary_rate = stationary_env.step(stationary_actions(stationary_env))[4]["rate_e2e_bps"]
    equal_env = copy.deepcopy(env)
    equal_rate = equal_env.step(equal_spacing_actions(equal_env))[4]["rate_e2e_bps"]
    positions_before = np.stack([state.position_m for state in env.states]).copy()
    velocities_before = np.stack([state.velocity_mps for state in env.states]).copy()
    step_before = env.step_index
    previous_applied_before = env._last_applied_relay_velocities.copy()
    actions = greedy_one_step_actions(env, sweeps=2)
    positions_after = np.stack([state.position_m for state in env.states])
    velocities_after = np.stack([state.velocity_mps for state in env.states])
    assert np.array_equal(positions_after, positions_before)
    assert np.array_equal(velocities_after, velocities_before)
    assert env.step_index == step_before
    assert np.array_equal(env._last_applied_relay_velocities, previous_applied_before)
    _, _, terminated, _, info = env.step(actions)
    assert not terminated
    expected_minimum = max(stationary_rate, equal_rate)
    assert info["rate_e2e_bps"] + 1e-9 >= expected_minimum


def test_baselines_run_for_50_steps_without_nan_or_unhandled_errors() -> None:
    for baseline in (
        stationary_actions,
        equal_spacing_actions,
        weighted_spacing_actions,
        greedy_one_step_actions,
    ):
        env = MultiRelayEnvironment()
        env.reset(seed=0)
        for attempt in range(50):
            actions = baseline(env)
            assert actions.shape == (env.config.num_relays, 3)
            assert np.all(np.isfinite(actions))
            assert np.all(actions >= -1.0) and np.all(actions <= 1.0)
            _, _, terminated, truncated, info = env.step(actions)
            assert np.all(np.isfinite(info["positions_m"]))
            assert np.all(np.isfinite(info["velocities_mps"]))
            assert np.all(np.isfinite(info["hop_capacities_bps"]))
            assert np.isfinite(info["rate_e2e_bps"])
            assert info["rate_e2e_bps"] >= 0.0
            if truncated:
                env.reset(seed=attempt + 1)
            elif terminated:
                env.reset(seed=attempt + 1)


def test_random_actions_run_for_500_steps_without_nan_or_unhandled_failure() -> None:
    env = MultiRelayEnvironment()
    rng = np.random.default_rng(3)
    env.reset(seed=3)
    for _ in range(500):
        _, _, terminated, truncated, info = env.step(rng.uniform(-1.0, 1.0, (4, 3)))
        assert np.all(np.isfinite(info["positions_m"]))
        assert np.all(np.isfinite(info["velocities_mps"]))
        assert np.all(np.isfinite(info["hop_capacities_bps"]))
        if terminated or truncated:
            env.reset(seed=3)
