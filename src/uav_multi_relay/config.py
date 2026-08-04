"""Validated configuration data for the multi-relay environment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import MotionLimits


def _finite_vector3(value: object, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    result = vector.copy()
    result.setflags(write=False)
    return result


def _positive_finite(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return float(value)


@dataclass(frozen=True)
class FlightBounds:
    """Closed three-dimensional bounds in which UAVs may fly."""

    minimum_m: np.ndarray
    maximum_m: np.ndarray

    def __post_init__(self) -> None:
        minimum = _finite_vector3(self.minimum_m, "minimum_m")
        maximum = _finite_vector3(self.maximum_m, "maximum_m")
        if np.any(minimum >= maximum):
            raise ValueError("each minimum_m component must be less than maximum_m")
        object.__setattr__(self, "minimum_m", minimum)
        object.__setattr__(self, "maximum_m", maximum)

    def contains(self, position_m: object) -> bool:
        """Return whether a finite position is within the closed flight bounds."""
        position = _finite_vector3(position_m, "position_m")
        return bool(np.all(position >= self.minimum_m) and np.all(position <= self.maximum_m))


@dataclass(frozen=True)
class ChannelConfig:
    """Physical parameters for the first air-to-air channel model."""

    carrier_frequency_hz: float
    reference_distance_m: float
    path_loss_exponent: float
    bandwidth_hz: float
    transmit_power_w: float
    noise_psd_w_per_hz: float
    noise_figure_linear: float
    maximum_antenna_gain_linear: float
    minimum_antenna_gain_linear: float
    minimum_distance_m: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            object.__setattr__(self, name, _positive_finite(value, name))
        if self.minimum_antenna_gain_linear > self.maximum_antenna_gain_linear:
            raise ValueError("minimum_antenna_gain_linear must not exceed maximum")

    @property
    def reference_gain_linear(self) -> float:
        """Free-space reference gain, excluding directional antenna gains."""
        speed_of_light_mps = 299_792_458.0
        return float(
            (
                speed_of_light_mps
                / (4.0 * np.pi * self.carrier_frequency_hz * self.reference_distance_m)
            )
            ** 2
        )


@dataclass(frozen=True)
class EndpointTrajectoryConfig:
    """Altitude and waypoint parameters for one endpoint UAV."""

    altitude_min_m: float
    altitude_max_m: float
    waypoint_radius_m: float
    waypoint_count: int
    arrival_tolerance_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "altitude_min_m", float(self.altitude_min_m))
        object.__setattr__(self, "altitude_max_m", float(self.altitude_max_m))
        object.__setattr__(self, "waypoint_radius_m", float(self.waypoint_radius_m))
        object.__setattr__(self, "arrival_tolerance_m", float(self.arrival_tolerance_m))
        if not all(
            np.isfinite(value)
            for value in (
                self.altitude_min_m,
                self.altitude_max_m,
                self.waypoint_radius_m,
                self.arrival_tolerance_m,
            )
        ):
            raise ValueError("endpoint trajectory values must be finite")
        if self.altitude_min_m >= self.altitude_max_m:
            raise ValueError("altitude_min_m must be less than altitude_max_m")
        if self.waypoint_radius_m <= self.arrival_tolerance_m or self.arrival_tolerance_m <= 0:
            raise ValueError("waypoint_radius_m must be greater than arrival_tolerance_m > 0")
        if (
            isinstance(self.waypoint_count, bool)
            or not isinstance(self.waypoint_count, int)
            or self.waypoint_count < 2
        ):
            raise ValueError("waypoint_count must be an integer of at least 2")


def _default_high_trajectory() -> EndpointTrajectoryConfig:
    return EndpointTrajectoryConfig(170.0, 230.0, 30.0, 4, 2.0)


def _default_low_trajectory() -> EndpointTrajectoryConfig:
    return EndpointTrajectoryConfig(50.0, 110.0, 30.0, 4, 2.0)


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration for a synchronous multi-relay simulation episode."""

    num_relays: int
    delta_t_s: float
    max_steps: int
    relay_motion_limits: MotionLimits
    high_motion_limits: MotionLimits
    low_motion_limits: MotionLimits
    flight_bounds: FlightBounds
    hard_safety_distance_m: float
    soft_safety_distance_m: float
    hard_max_link_distance_m: float
    rate_reference_bps: float
    channel: ChannelConfig
    high_trajectory: EndpointTrajectoryConfig = field(default_factory=_default_high_trajectory)
    low_trajectory: EndpointTrajectoryConfig = field(default_factory=_default_low_trajectory)

    def __post_init__(self) -> None:
        if isinstance(self.num_relays, bool) or not isinstance(self.num_relays, int) or self.num_relays < 1:
            raise ValueError("num_relays must be a positive integer")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        for name in (
            "delta_t_s",
            "hard_safety_distance_m",
            "soft_safety_distance_m",
            "hard_max_link_distance_m",
            "rate_reference_bps",
        ):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name))
        if self.soft_safety_distance_m < self.hard_safety_distance_m:
            raise ValueError("soft_safety_distance_m must be at least hard_safety_distance_m")
        if self.hard_max_link_distance_m < self.hard_safety_distance_m:
            raise ValueError(
                "hard_max_link_distance_m must be at least hard_safety_distance_m"
            )
        if not all(
            isinstance(value, MotionLimits)
            for value in (
                self.relay_motion_limits,
                self.high_motion_limits,
                self.low_motion_limits,
            )
        ):
            raise ValueError("motion limits must be MotionLimits instances")
        if not isinstance(self.flight_bounds, FlightBounds):
            raise ValueError("flight_bounds must be a FlightBounds instance")
        if not isinstance(self.channel, ChannelConfig):
            raise ValueError("channel must be a ChannelConfig instance")
        if not isinstance(self.high_trajectory, EndpointTrajectoryConfig):
            raise ValueError("high_trajectory must be an EndpointTrajectoryConfig instance")
        if not isinstance(self.low_trajectory, EndpointTrajectoryConfig):
            raise ValueError("low_trajectory must be an EndpointTrajectoryConfig instance")
        lower_altitude = self.flight_bounds.minimum_m[2]
        upper_altitude = self.flight_bounds.maximum_m[2]
        for name, trajectory in (
            ("high_trajectory", self.high_trajectory),
            ("low_trajectory", self.low_trajectory),
        ):
            if (
                trajectory.altitude_min_m < lower_altitude
                or trajectory.altitude_max_m > upper_altitude
            ):
                raise ValueError(f"{name} altitude range must be within flight bounds")
        if self.high_trajectory.altitude_min_m <= self.low_trajectory.altitude_max_m:
            raise ValueError("high_trajectory altitude range must be strictly above low_trajectory")


def default_environment_config() -> EnvironmentConfig:
    """Create the deterministic four-relay configuration used by default."""
    return EnvironmentConfig(
        num_relays=4,
        delta_t_s=0.2,
        max_steps=500,
        relay_motion_limits=MotionLimits(30.0, 12.0, 12.0, 15.0, 8.0),
        high_motion_limits=MotionLimits(20.0, 10.0, 10.0, 10.0, 5.0),
        low_motion_limits=MotionLimits(20.0, 10.0, 10.0, 10.0, 5.0),
        flight_bounds=FlightBounds(
            np.array([-500.0, -500.0, 30.0]), np.array([500.0, 500.0, 250.0])
        ),
        hard_safety_distance_m=15.0,
        soft_safety_distance_m=35.0,
        hard_max_link_distance_m=250.0,
        rate_reference_bps=10_000_000.0,
        channel=ChannelConfig(
            carrier_frequency_hz=2.4e9,
            reference_distance_m=1.0,
            path_loss_exponent=2.2,
            bandwidth_hz=20_000_000.0,
            transmit_power_w=0.1,
            noise_psd_w_per_hz=3.98e-21,
            noise_figure_linear=3.16,
            maximum_antenna_gain_linear=1.5,
            minimum_antenna_gain_linear=0.1,
            minimum_distance_m=1.0,
        ),
        high_trajectory=_default_high_trajectory(),
        low_trajectory=_default_low_trajectory(),
    )
