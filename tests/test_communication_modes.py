import numpy as np
import pytest

from uav_multi_relay import MultiRelayEnvironment, scenario_environment_config


def test_default_communication_modes_match_explicit_defaults():
    default = MultiRelayEnvironment()
    explicit = MultiRelayEnvironment(scenario_environment_config(default.config, tdma_mode="optimal", antenna_mode="dipole"))
    default.reset(seed=11)
    explicit.reset(seed=11)
    default_info = default.step(np.zeros((default.config.num_relays, 3)))[4]
    explicit_info = explicit.step(np.zeros((explicit.config.num_relays, 3)))[4]
    assert default.config.tdma_mode == "optimal" and default.config.antenna_mode == "dipole"
    assert default_info["rate_e2e_bps"] == pytest.approx(explicit_info["rate_e2e_bps"])


def test_equal_tdma_and_isotropic_modes_are_valid_and_observable():
    base = MultiRelayEnvironment().config
    equal = MultiRelayEnvironment(scenario_environment_config(base, tdma_mode="equal", antenna_mode="dipole"))
    isotropic = MultiRelayEnvironment(scenario_environment_config(base, tdma_mode="optimal", antenna_mode="isotropic"))
    equal.reset(seed=4)
    isotropic.reset(seed=4)
    equal_info = equal.step(np.zeros((equal.config.num_relays, 3)))[4]
    isotropic_info = isotropic.step(np.zeros((isotropic.config.num_relays, 3)))[4]
    assert equal_info["tdma_fractions"].sum() == pytest.approx(1.0)
    assert all(np.isfinite(value) for value in (equal_info["rate_e2e_bps"], isotropic_info["rate_e2e_bps"]))
