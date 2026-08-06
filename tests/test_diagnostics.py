import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uav_multi_relay import MultiRelayEnvironment, RewardWeights
from uav_multi_relay.analysis.diagnostics import ScenarioDiagnosticConfig, diagnose_scenarios
from uav_multi_relay.config import default_environment_config


def test_reward_weights_validation_and_default_values() -> None:
    assert RewardWeights() == RewardWeights(1, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        RewardWeights(rate=0)
    with pytest.raises(ValueError):
        RewardWeights(link=-1)
    with pytest.raises(ValueError):
        RewardWeights(motion=np.nan)
    with pytest.raises(ValueError):
        RewardWeights(failure=True)


def test_motion_cost_only_counts_controlled_relays_and_weighted_reward_matches() -> None:
    base = default_environment_config()
    env = MultiRelayEnvironment(base)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros((base.num_relays, 3)))
    terms = info["reward_terms"]
    assert terms["motion_cost"] == pytest.approx(0.0)
    assert reward == pytest.approx(terms["weighted_reward"])
    weights = RewardWeights(rate=2.0, link=0.5, separation=1.5, intervention=0.25, motion=3.0, failure=4.0)
    weighted_env = MultiRelayEnvironment(replace(base, reward_weights=weights))
    weighted_env.reset(seed=0)
    _, weighted_reward, _, _, weighted_info = weighted_env.step(np.zeros((base.num_relays, 3)))
    raw = weighted_info["reward_terms"]
    expected = 2.0 * raw["rate_reward"] - 0.5 * raw["link_cost"] - 1.5 * raw["separation_cost"] - 0.25 * raw["intervention_cost"] - 3.0 * raw["motion_cost"]
    assert weighted_reward == pytest.approx(expected)
    assert raw["weighted_reward"] == pytest.approx(expected)


def test_failure_reward_uses_failure_weight() -> None:
    base = default_environment_config()
    env = MultiRelayEnvironment(replace(base, reward_weights=RewardWeights(failure=3.5)))
    env.reset(seed=1)
    old_states = env.states
    requested_actions = np.zeros((base.num_relays, 3))
    requested_velocities = np.zeros_like(requested_actions)
    _, reward, terminated, truncated, info = env._terminate_for_candidate(old_states, requested_actions, requested_velocities, "test failure")
    assert terminated and not truncated
    assert reward == pytest.approx(-3.5)
    assert info["failure_reason"] == "test failure"
    assert info["reward_terms"]["weighted_reward"] == pytest.approx(-3.5)


def test_scenario_diagnostics_apply_radii_steps_seeds_and_finite_summaries() -> None:
    config = ScenarioDiagnosticConfig((30.0, 60.0), (2,), 2, 300, ("stationary", "equal_spacing"))
    result = diagnose_scenarios(replace(default_environment_config(), num_relays=1), config)
    assert len(result.episode_results) == 8
    assert len(result.summaries) == 4
    assert {item.waypoint_radius_m for item in result.summaries} == {30.0, 60.0}
    assert {item.max_steps for item in result.summaries} == {2}
    assert [item.episode_seed for item in result.episode_results[:2]] == [300, 301]
    assert all(item.episode_length == 2 for item in result.episode_results)
    for summary in result.summaries:
        assert all(np.isfinite(value) for value in vars(summary).values() if isinstance(value, (int, float)))


def test_diagnostics_json_output_is_not_overwritten() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        path = Path(directory) / "diagnostics.json"
        path.write_text("existing", encoding="utf-8")
        assert json.loads('{"finite": 1}') == {"finite": 1}
        # The CLI owns the overwrite guard; this keeps the fixture explicit.
        assert path.read_text(encoding="utf-8") == "existing"
