import math

import numpy as np
import pytest

from uav_multi_relay.communication import (
    channel_power_gain,
    compute_chain_geometries,
    compute_link_geometry,
    dipole_gain,
    equal_tdma_rate,
    optimal_tdma_rate,
    ordered_nodes,
    shannon_capacity_bps,
    snr_linear,
)
from uav_multi_relay.core import MotionLimits, UAVState
from uav_multi_relay.kinematics import _speed_limit_tolerance, advance_state, make_velocity_feasible
from uav_multi_relay.safety import velocity_to_normalized_action


@pytest.fixture
def limits() -> MotionLimits:
    return MotionLimits(5.0, 2.0, 3.0, 2.0, 1.0)


def test_state_is_an_immutable_copy() -> None:
    position = np.array([1.0, 2.0, 3.0])
    state = UAVState("uav", position, np.zeros(3))
    position[0] = 99.0
    assert state.position_m[0] == 1.0
    with pytest.raises(ValueError):
        state.position_m[0] = 4.0


@pytest.mark.parametrize("bad_vector", [np.zeros(2), [0.0, 0.0, np.nan]])
def test_state_rejects_invalid_vectors(bad_vector: object) -> None:
    with pytest.raises(ValueError):
        UAVState("uav", bad_vector, np.zeros(3))


def test_velocity_is_limited_by_horizontal_speed(limits: MotionLimits) -> None:
    applied = make_velocity_feasible([100.0, 100.0, 0.0], [0.0, 0.0, 0.0], limits, 10.0)
    assert np.linalg.norm(applied[:2]) <= limits.max_horizontal_speed_mps


def test_velocity_has_separate_climb_and_descent_limits(limits: MotionLimits) -> None:
    climbing = make_velocity_feasible([0.0, 0.0, 50.0], [0.0, 0.0, 0.0], limits, 10.0)
    descending = make_velocity_feasible([0.0, 0.0, -50.0], [0.0, 0.0, 0.0], limits, 10.0)
    assert climbing[2] == limits.max_climb_speed_mps
    assert descending[2] == -limits.max_descent_speed_mps


def test_velocity_change_respects_acceleration(limits: MotionLimits) -> None:
    current = np.array([1.0, 0.0, 0.5])
    applied = make_velocity_feasible([20.0, 20.0, 20.0], current, limits, 0.5)
    assert np.linalg.norm(applied[:2] - current[:2]) <= 1.0
    assert abs(applied[2] - current[2]) <= 0.5


@pytest.mark.parametrize(
    "current",
    [[5.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, -3.0]],
)
def test_velocity_accepts_current_values_exactly_at_limits(
    limits: MotionLimits, current: object
) -> None:
    applied = make_velocity_feasible(current, current, limits, 1.0)
    assert np.linalg.norm(applied[:2]) <= limits.max_horizontal_speed_mps
    assert -limits.max_descent_speed_mps <= applied[2] <= limits.max_climb_speed_mps


@pytest.mark.parametrize(
    "current",
    [
        [np.nextafter(5.0, np.inf), 0.0, 0.0],
        [0.0, 0.0, np.nextafter(2.0, np.inf)],
        [0.0, 0.0, np.nextafter(-3.0, -np.inf)],
    ],
)
def test_velocity_normalizes_one_ulp_boundary_excess(
    limits: MotionLimits, current: object
) -> None:
    applied = make_velocity_feasible(current, current, limits, 1.0)
    assert np.linalg.norm(applied[:2]) <= limits.max_horizontal_speed_mps
    assert -limits.max_descent_speed_mps <= applied[2] <= limits.max_climb_speed_mps


def test_velocity_normalizes_reported_horizontal_boundary_rounding() -> None:
    limits = MotionLimits(30.0, 12.0, 12.0, 15.0, 8.0)
    applied = make_velocity_feasible(
        [30.0, 0.0, 0.0], [30.000000000000004, 0.0, 0.0], limits, 0.2
    )
    assert np.linalg.norm(applied[:2]) <= limits.max_horizontal_speed_mps


def test_speed_limit_tolerance_is_the_shared_machine_precision_rule() -> None:
    limit = 30.0
    assert _speed_limit_tolerance(limit) == pytest.approx(
        64.0 * np.finfo(float).eps * max(1.0, abs(limit))
    )


@pytest.mark.parametrize(
    "velocity",
    [
        [30.000000000000004, 0.0, 0.0],
        [np.nextafter(30.0, np.inf), 0.0, 0.0],
        [0.0, 0.0, np.nextafter(12.0, np.inf)],
        [0.0, 0.0, np.nextafter(-12.0, -np.inf)],
    ],
)
def test_velocity_entry_points_share_tolerance_for_one_ulp_boundary_values(
    velocity: object,
) -> None:
    limits = MotionLimits(30.0, 12.0, 12.0, 15.0, 8.0)
    applied = make_velocity_feasible(velocity, velocity, limits, 1.0)
    action = velocity_to_normalized_action(velocity, limits)
    assert np.linalg.norm(applied[:2]) <= limits.max_horizontal_speed_mps
    assert -limits.max_descent_speed_mps <= applied[2] <= limits.max_climb_speed_mps
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0)


@pytest.mark.parametrize(
    "velocity",
    [[30.0 + 1e-6, 0.0, 0.0], [0.0, 0.0, 12.0 + 1e-6], [0.0, 0.0, -12.0 - 1e-6]],
)
def test_velocity_entry_points_reject_values_beyond_shared_tolerance(velocity: object) -> None:
    limits = MotionLimits(30.0, 12.0, 12.0, 15.0, 8.0)
    with pytest.raises(ValueError):
        make_velocity_feasible([0.0, 0.0, 0.0], velocity, limits, 1.0)
    with pytest.raises(ValueError):
        velocity_to_normalized_action(velocity, limits)


@pytest.mark.parametrize(
    "current",
    [[5.0 + 1e-6, 0.0, 0.0], [0.0, 0.0, 2.0 + 1e-6], [0.0, 0.0, -3.0 - 1e-6]],
)
def test_velocity_rejects_current_values_beyond_boundary_tolerance(
    limits: MotionLimits, current: object
) -> None:
    with pytest.raises(ValueError, match="speed limits"):
        make_velocity_feasible([0.0, 0.0, 0.0], current, limits, 1.0)


def test_velocity_return_value_is_strictly_within_limits(limits: MotionLimits) -> None:
    applied = make_velocity_feasible(
        [100.0, -100.0, 100.0], [np.nextafter(5.0, np.inf), 0.0, -3.0], limits, 1.0
    )
    assert np.linalg.norm(applied[:2]) <= limits.max_horizontal_speed_mps
    assert -limits.max_descent_speed_mps <= applied[2] <= limits.max_climb_speed_mps


def test_repeated_boundary_motion_does_not_accumulate_an_invalid_velocity() -> None:
    limits = MotionLimits(30.0, 12.0, 12.0, 15.0, 8.0)
    current = np.array([30.000000000000004, 0.0, 0.0])
    for _ in range(100):
        current = make_velocity_feasible([30.0, 0.0, 0.0], current, limits, 0.2)
        assert np.linalg.norm(current[:2]) <= limits.max_horizontal_speed_mps
        assert -limits.max_descent_speed_mps <= current[2] <= limits.max_climb_speed_mps


@pytest.mark.parametrize("current", [[6.0, 0.0, 0.0], [0.0, 0.0, 2.1], [0.0, 0.0, -3.1]])
def test_velocity_rejects_infeasible_current_velocity(
    limits: MotionLimits, current: object
) -> None:
    with pytest.raises(ValueError):
        make_velocity_feasible([0.0, 0.0, 0.0], current, limits, 1.0)


def test_advance_state_uses_applied_velocity_without_mutating_input() -> None:
    state = UAVState("uav", [1.0, 2.0, 3.0], [9.0, 9.0, 9.0])
    advanced = advance_state(state, [2.0, -1.0, 0.5], 4.0)
    assert np.allclose(advanced.position_m, [9.0, -2.0, 5.0])
    assert np.allclose(advanced.velocity_mps, [2.0, -1.0, 0.5])
    assert np.allclose(state.position_m, [1.0, 2.0, 3.0])


def test_chain_geometry_for_four_relays() -> None:
    nodes = ordered_nodes(
        UAVState("H", [0.0, 0.0, 0.0], np.zeros(3)),
        tuple(UAVState(f"R{i}", [float(i), 0.0, 0.0], np.zeros(3)) for i in range(1, 5)),
        UAVState("L", [5.0, 0.0, 0.0], np.zeros(3)),
    )
    assert len(nodes) == 6
    assert len(compute_chain_geometries(nodes)) == 5


def test_vertical_link_geometry_is_stable() -> None:
    geometry = compute_link_geometry([0.0, 0.0, 0.0], [0.0, 0.0, 10.0])
    assert geometry.horizontal_distance_m == 0.0
    assert geometry.elevation_angle_rad == pytest.approx(math.pi / 2)


def test_dipole_gain_has_a_floor() -> None:
    assert dipole_gain(math.pi / 2, 10.0, 0.2) == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("elevation_angle_rad", "max_gain_linear", "min_gain_linear"),
    [(-0.1, 10.0, 0.2), (math.pi / 2 + 0.1, 10.0, 0.2), (0.0, 0.2, 10.0)],
)
def test_dipole_gain_rejects_invalid_parameters(
    elevation_angle_rad: float, max_gain_linear: float, min_gain_linear: float
) -> None:
    with pytest.raises(ValueError):
        dipole_gain(elevation_angle_rad, max_gain_linear, min_gain_linear)


def test_channel_gain_applies_reference_gain_and_antennas_once() -> None:
    gain = channel_power_gain(10.0, 2.0, 1.0, 2.0, 3.0, 5.0, 1.0)
    assert gain == pytest.approx(2.0 * (10.0**-2.0) * 3.0 * 5.0)


def test_snr_and_capacity_are_finite_and_nonnegative() -> None:
    snr = snr_linear(1.0, 0.5, 1e-9, 1e6, 2.0)
    capacity = shannon_capacity_bps(1e6, snr)
    assert math.isfinite(snr) and snr >= 0.0
    assert math.isfinite(capacity) and capacity >= 0.0


def test_equal_tdma_rate() -> None:
    rate, fractions = equal_tdma_rate([10.0, 20.0, 30.0, 40.0, 50.0])
    assert rate == 2.0
    assert np.allclose(fractions, 0.2)


def test_optimal_tdma_equalizes_hop_throughput() -> None:
    capacities = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    rate, fractions = optimal_tdma_rate(capacities)
    expected_rate = 1.0 / np.sum(1.0 / capacities)
    assert rate == pytest.approx(expected_rate)
    assert np.sum(fractions) == pytest.approx(1.0)
    assert fractions * capacities == pytest.approx(np.full_like(capacities, rate))


@pytest.mark.parametrize("tdma", [equal_tdma_rate, optimal_tdma_rate])
def test_tdma_handles_zero_capacity(tdma: object) -> None:
    rate, fractions = tdma([10.0, 0.0, 30.0])
    assert rate == 0.0
    assert len(fractions) == 3
    assert np.all(np.isfinite(fractions))
