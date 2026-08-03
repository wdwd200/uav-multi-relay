"""Geometry, channel, capacity, and TDMA calculations for relay chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import UAVState, _vector3


def _positive_finite(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return float(value)


@dataclass(frozen=True)
class LinkGeometry:
    """Three-dimensional geometry of a directed communication link."""

    distance_3d_m: float
    horizontal_distance_m: float
    elevation_angle_rad: float


def ordered_nodes(
    high: UAVState, relays: tuple[UAVState, ...], low: UAVState
) -> tuple[UAVState, ...]:
    """Return the ordered H-to-L node sequence, requiring at least one relay."""
    nodes = (high, *relays, low)
    if not relays:
        raise ValueError("at least one relay is required")
    if len({node.name for node in nodes}) != len(nodes):
        raise ValueError("node names must be unique")
    return nodes


def compute_link_geometry(tx_position_m: object, rx_position_m: object) -> LinkGeometry:
    """Compute distance, horizontal distance, and nonnegative elevation angle."""
    difference = _vector3(rx_position_m, "rx_position_m") - _vector3(
        tx_position_m, "tx_position_m"
    )
    horizontal = float(np.linalg.norm(difference[:2]))
    distance = float(np.linalg.norm(difference))
    elevation = float(np.arctan2(abs(difference[2]), horizontal))
    return LinkGeometry(distance, horizontal, elevation)


def compute_chain_geometries(nodes: tuple[UAVState, ...]) -> tuple[LinkGeometry, ...]:
    """Compute each ordered adjacent-link geometry in a node chain."""
    if len(nodes) < 2:
        raise ValueError("at least two nodes are required")
    return tuple(
        compute_link_geometry(tx.position_m, rx.position_m)
        for tx, rx in zip(nodes, nodes[1:])
    )


def dipole_gain(
    elevation_angle_rad: float, max_gain_linear: float, min_gain_linear: float
) -> float:
    """Return the simplified vertical short-dipole gain in linear units."""
    if not np.isfinite(elevation_angle_rad):
        raise ValueError("elevation_angle_rad must be finite")
    maximum = _positive_finite(max_gain_linear, "max_gain_linear")
    minimum = _positive_finite(min_gain_linear, "min_gain_linear")
    return max(minimum, maximum * float(np.cos(elevation_angle_rad) ** 2))


def channel_power_gain(
    distance_m: float,
    reference_gain_linear: float,
    reference_distance_m: float,
    path_loss_exponent: float,
    tx_gain_linear: float,
    rx_gain_linear: float,
    minimum_distance_m: float,
) -> float:
    """Return path loss and directional antenna gain as one linear power gain."""
    if not np.isfinite(distance_m) or distance_m < 0:
        raise ValueError("distance_m must be finite and nonnegative")
    reference_gain = _positive_finite(reference_gain_linear, "reference_gain_linear")
    reference_distance = _positive_finite(reference_distance_m, "reference_distance_m")
    exponent = _positive_finite(path_loss_exponent, "path_loss_exponent")
    tx_gain = _positive_finite(tx_gain_linear, "tx_gain_linear")
    rx_gain = _positive_finite(rx_gain_linear, "rx_gain_linear")
    effective_distance = max(float(distance_m), _positive_finite(minimum_distance_m, "minimum_distance_m"))
    return reference_gain * (effective_distance / reference_distance) ** (-exponent) * tx_gain * rx_gain


def snr_linear(
    transmit_power_w: float,
    channel_gain_linear: float,
    noise_psd_w_per_hz: float,
    bandwidth_hz: float,
    noise_figure_linear: float,
) -> float:
    """Calculate linear SNR from transmit power, gain, and receiver noise."""
    if not np.isfinite(transmit_power_w) or transmit_power_w < 0:
        raise ValueError("transmit_power_w must be finite and nonnegative")
    if not np.isfinite(channel_gain_linear) or channel_gain_linear < 0:
        raise ValueError("channel_gain_linear must be finite and nonnegative")
    denominator = (
        _positive_finite(noise_psd_w_per_hz, "noise_psd_w_per_hz")
        * _positive_finite(bandwidth_hz, "bandwidth_hz")
        * _positive_finite(noise_figure_linear, "noise_figure_linear")
    )
    return float(transmit_power_w * channel_gain_linear / denominator)


def shannon_capacity_bps(bandwidth_hz: float, snr_value_linear: float) -> float:
    """Calculate Shannon capacity in bits per second."""
    bandwidth = _positive_finite(bandwidth_hz, "bandwidth_hz")
    if not np.isfinite(snr_value_linear) or snr_value_linear < 0:
        raise ValueError("snr_value_linear must be finite and nonnegative")
    return float(bandwidth * np.log2(1.0 + snr_value_linear))


def _capacities(capacities_bps: object) -> np.ndarray:
    capacities = np.asarray(capacities_bps, dtype=float)
    if capacities.ndim != 1 or capacities.size == 0 or not np.all(np.isfinite(capacities)):
        raise ValueError("capacities_bps must be a non-empty, finite one-dimensional array")
    return capacities


def equal_tdma_rate(capacities_bps: object) -> tuple[float, np.ndarray]:
    """Calculate equal-slot TDMA throughput and time fractions."""
    capacities = _capacities(capacities_bps)
    fractions = np.full(capacities.size, 1.0 / capacities.size)
    if np.any(capacities <= 0):
        return 0.0, fractions
    return float(np.min(capacities) / capacities.size), fractions


def optimal_tdma_rate(capacities_bps: object) -> tuple[float, np.ndarray]:
    """Calculate analytical optimal TDMA throughput and time fractions."""
    capacities = _capacities(capacities_bps)
    if np.any(capacities <= 0):
        return 0.0, np.full(capacities.size, 1.0 / capacities.size)
    inverse_capacities = 1.0 / capacities
    inverse_sum = float(np.sum(inverse_capacities))
    return float(1.0 / inverse_sum), inverse_capacities / inverse_sum
