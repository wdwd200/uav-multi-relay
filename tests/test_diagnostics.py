import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnose_masac import _contributions, _run_policy, _update_scale_summary
from uav_multi_relay import MultiRelayEnvironment, RewardWeights
from uav_multi_relay.analysis.diagnostics import ScenarioDiagnosticConfig, diagnose_scenarios
from uav_multi_relay.config import default_environment_config
from uav_multi_relay.learning import MultiAgentReplayBuffer, ParameterSharingMASAC
from uav_multi_relay.training import MASACExperimentConfig, MASACTrainingConfig, run_masac_experiment
from uav_multi_relay.training.trainer import _IntervalDiagnostics


def test_reward_weights_default_values_and_invalid_values() -> None:
    assert RewardWeights() == RewardWeights(1, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        RewardWeights(rate=0)
    with pytest.raises(ValueError):
        RewardWeights(link=-1)
    with pytest.raises(ValueError):
        RewardWeights(motion=np.nan)
    with pytest.raises(ValueError):
        RewardWeights(failure=True)


def test_motion_cost_uses_only_controlled_relays_and_weighted_reward_matches() -> None:
    base = default_environment_config()
    env = MultiRelayEnvironment(base)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros((base.num_relays, 3)))
    terms = info["reward_terms"]
    assert terms["motion_cost"] == pytest.approx(0.0)
    assert reward == pytest.approx(terms["weighted_reward"])
    weights = RewardWeights(rate=2.0, link=.5, separation=1.5, intervention=.25, motion=3.0, failure=4.0)
    weighted_env = MultiRelayEnvironment(replace(base, reward_weights=weights))
    weighted_env.reset(seed=0)
    _, weighted_reward, _, _, weighted_info = weighted_env.step(np.zeros((base.num_relays, 3)))
    raw = weighted_info["reward_terms"]
    expected = 2.0 * raw["rate_reward"] - .5 * raw["link_cost"] - 1.5 * raw["separation_cost"] - .25 * raw["intervention_cost"] - 3.0 * raw["motion_cost"]
    assert weighted_reward == pytest.approx(expected)
    assert raw["weighted_reward"] == pytest.approx(expected)
    moving_env = MultiRelayEnvironment(base)
    moving_env.reset(seed=0)
    _, _, _, _, moving_info = moving_env.step(np.ones((base.num_relays, 3)))
    assert moving_info["reward_terms"]["motion_cost"] > 0.0


def test_failure_reward_uses_failure_weight() -> None:
    base = default_environment_config()
    env = MultiRelayEnvironment(replace(base, reward_weights=RewardWeights(failure=3.5)))
    env.reset(seed=1)
    requested_actions = np.zeros((base.num_relays, 3))
    _, reward, terminated, truncated, info = env._terminate_for_candidate(
        env.states, requested_actions, np.zeros_like(requested_actions), "test failure"
    )
    assert terminated and not truncated
    assert reward == pytest.approx(-3.5)
    assert info["failure_reason"] == "test failure"
    assert info["reward_terms"]["weighted_reward"] == pytest.approx(-3.5)


@pytest.mark.parametrize("num_relays", [1, 4])
def test_scenario_diagnostics_cover_relay_counts_and_finite_summaries(num_relays: int) -> None:
    config = ScenarioDiagnosticConfig((30.0, 60.0), (2,), 2, 300, ("stationary", "equal_spacing"))
    result = diagnose_scenarios(replace(default_environment_config(), num_relays=num_relays), config)
    assert len(result.episode_results) == 8
    assert len(result.summaries) == 4
    assert {item.waypoint_radius_m for item in result.summaries} == {30.0, 60.0}
    assert [item.episode_seed for item in result.episode_results[:2]] == [300, 301]
    assert all(item.episode_length == 2 for item in result.episode_results)
    for summary in result.summaries:
        assert all(np.isfinite(value) for value in vars(summary).values() if isinstance(value, (int, float)))
        members = [item for item in result.episode_results if item.waypoint_radius_m == summary.waypoint_radius_m and item.policy == summary.policy]
        assert summary.minimum_rate_e2e_bps == min(item.min_rate_e2e_bps for item in members)
        assert summary.mean_episode_min_rate_e2e_bps == pytest.approx(np.mean([item.min_rate_e2e_bps for item in members]))


def test_scenario_diagnostic_json_output_is_not_overwritten() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        path = Path(directory) / "diagnostics.json"
        command = [sys.executable, "scripts/diagnose_scenarios.py", "--output", str(path), "--radii", "30", "--max-steps", "1", "--episodes", "1", "--seed", "0", "--policies", "stationary", "--num-relays", "1"]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        assert json.loads(completed.stdout)["episodes"] == 1
        assert "NaN" not in path.read_text(encoding="utf-8")
        assert subprocess.run(command, capture_output=True, text=True).returncode != 0


def test_interval_action_statistics_are_exact() -> None:
    stats = _IntervalDiagnostics()
    info = {
        "applied_relay_actions": np.array([[1.0, 0.0, 0.0], [0.0, .5, 0.0]]),
        "requested_relay_velocities_mps": np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        "applied_relay_velocities_mps": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "intervention_norms": np.array([1.0, 1.0]), "safety_scale": .5,
    }
    stats.add_step(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), info, False, False)
    stats.add_episode(2, 6.0)
    payload = stats.payload(1, 2)
    assert payload["action_mismatch_event_rate"] == 1.0
    assert payload["action_mismatch_l2_mean"] == pytest.approx(.25)
    assert payload["action_mismatch_l2_p95"] == pytest.approx(.475)
    assert payload["requested_action_saturation_rate"] == pytest.approx(2 / 6)
    assert payload["safety_scale_mean"] == payload["safety_scale_min"] == .5
    assert payload["intervention_event_rate"] == 1.0


def test_reward_contribution_matches_weighted_reward() -> None:
    env = MultiRelayEnvironment()
    terms = {"rate_reward": 3.0, "link_cost": .5, "separation_cost": .25, "intervention_cost": .2, "motion_cost": .1, "failure_penalty": 1.0, "weighted_reward": 0.0}
    weights = replace(env.config.reward_weights, intervention=.1, motion=.1)
    terms["weighted_reward"] = terms["rate_reward"] - terms["link_cost"] - terms["separation_cost"] - weights.intervention * terms["intervention_cost"] - weights.motion * terms["motion_cost"] - terms["failure_penalty"]
    result = _contributions([{"episode_length": 1, "episode_return": terms["weighted_reward"], "terminated": True, "truncated": False, "reward_terms": [terms]}], weights)
    assert result["weighted_total"] == pytest.approx(terms["weighted_reward"])


def test_periodic_checkpoints_and_diagnostic_failure_trace_are_written() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory); base = MultiRelayEnvironment(); env = MultiRelayEnvironment(replace(base.config, num_relays=1, max_steps=2))
        observation, _ = env.reset(seed=0)
        agent = ParameterSharingMASAC(observation["local"].shape[1], observation["global"].shape[0], 1, hidden_dims=(8, 8), device="cpu")
        replay = MultiAgentReplayBuffer(16, 1, observation["local"].shape[1], observation["global"].shape[0], 3, seed=0)
        config = MASACTrainingConfig(4, 16, 2, 1, 1, 1, 0)
        result = run_masac_experiment(env, MultiRelayEnvironment(env.config), agent, replay, config, MASACExperimentConfig(root / "run", 2, 2, 1, 10, 2, True))
        names = {path.name for path in (root / "run" / "checkpoints").iterdir()}
        assert {"step_000000.pt", "step_000002.pt", "step_000004.pt"} <= names
        assert result.final_checkpoint.is_file() and result.best_checkpoint.is_file()
        assert (root / "run" / "failure_traces.jsonl").is_file()
        assert "NaN" not in (root / "run" / "failure_traces.jsonl").read_text(encoding="utf-8")


def test_evaluation_applied_q_is_not_named_as_replay_q() -> None:
    base = MultiRelayEnvironment()
    env = MultiRelayEnvironment(replace(base.config, num_relays=1, max_steps=1))
    observation, _ = env.reset(seed=0)
    agent = ParameterSharingMASAC(observation["local"].shape[1], observation["global"].shape[0], 1, hidden_dims=(8, 8), device="cpu")
    _, diagnostics = _run_policy(env, agent, "masac", 1, 0, detailed=True)
    assert "evaluation_applied_action_q_mean" in diagnostics
    assert "actor_raw_minus_evaluation_applied_q_mean" in diagnostics
    assert "replay_applied_action_q_mean" not in diagnostics
    assert "actor_raw_minus_replay_q_mean" not in diagnostics


def test_update_scale_summary_reports_real_log_shapes_and_rejects_nonfinite() -> None:
    fields = ("actor_gradient_norm", "critic_gradient_norm", "critic_loss", "td_error_mean", "td_error_p95", "td_error_max", "q1_mean", "q2_mean", "target_q_mean", "alpha")
    first = {"environment_steps": 2, **{field: 1.0 for field in fields}}
    last = {"environment_steps": 4, **{field: 3.0 for field in fields}}
    summary = _update_scale_summary([first, last])
    assert summary["actor_gradient_norm"] == {"first_step": 2, "first_value": 1.0, "max_step": 4, "max_value": 3.0, "last_step": 4, "last_value": 3.0, "all_finite": True, "monotonic_non_decreasing": True, "single_point_spike": False}
    broken = dict(last)
    broken["td_error_mean"] = float("nan")
    with pytest.raises(ValueError, match="non-finite td_error_mean"):
        _update_scale_summary([first, broken])
