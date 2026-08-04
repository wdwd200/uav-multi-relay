import numpy as np
import pytest

from uav_multi_relay.baselines import equal_spacing_actions
from uav_multi_relay.config import (
    ChannelConfig,
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


def test_waypoint_follower_cycles_and_resets() -> None:
    follower = WaypointFollower([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    limits = MotionLimits(5.0, 2.0, 2.0, 10.0, 10.0)
    state = UAVState("H", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    velocity = follower.velocity_for(state, limits, 1.0)
    assert velocity[0] > 0.0
    follower.reset()
    assert follower.velocity_for(state, limits, 1.0)[0] > 0.0


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
    assert np.allclose(
        info["requested_relay_velocities_mps"], actions * env.config.relay_motion_limits.max_horizontal_speed_mps
    )


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
