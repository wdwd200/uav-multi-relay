import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnose_masac import _contributions
from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.learning import MultiAgentReplayBuffer, ParameterSharingMASAC
from uav_multi_relay.training import MASACExperimentConfig, MASACTrainingConfig, run_masac_experiment
from uav_multi_relay.training.trainer import _IntervalDiagnostics


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
