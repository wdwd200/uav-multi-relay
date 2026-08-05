import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.learning import MultiAgentReplayBuffer, ParameterSharingMASAC
from uav_multi_relay.training import (
    MASACExperimentConfig,
    MASACTrainingConfig,
    load_masac_checkpoint,
    run_masac_experiment,
)


def _experiment_parts(directory: Path, relays: int = 1):
    base = MultiRelayEnvironment()
    training_env = MultiRelayEnvironment(replace(base.config, num_relays=relays, max_steps=2))
    evaluation_env = MultiRelayEnvironment(replace(base.config, num_relays=relays, max_steps=2))
    observation, _ = training_env.reset(seed=0)
    agent = ParameterSharingMASAC(observation["local"].shape[1], observation["global"].shape[0], relays, hidden_dims=(8, 8), device="cpu")
    replay = MultiAgentReplayBuffer(8, relays, observation["local"].shape[1], observation["global"].shape[0], 3, seed=0)
    training = MASACTrainingConfig(3, 8, 2, 1, 1, 1, 0)
    experiment = MASACExperimentConfig(directory, 2, 2, 1, 100)
    return training_env, evaluation_env, agent, replay, training, experiment


def test_experiment_writes_finite_artifacts_and_metadata() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        output = Path(directory) / "run"
        parts = _experiment_parts(output)
        result = run_masac_experiment(*parts)
        expected = {"run_config.json", "training_metrics.jsonl", "evaluation_metrics.jsonl", "best_checkpoint.pt", "final_checkpoint.pt", "summary.json"}
        assert {path.name for path in output.iterdir()} == expected
        config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        assert config["observation_dimensions"]["local"] == [1, 23]
        training_lines = [json.loads(line) for line in (output / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        evaluation_lines = [json.loads(line) for line in (output / "evaluation_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [line["environment_steps"] for line in training_lines] == [2, 3]
        assert [line["environment_steps"] for line in evaluation_lines] == [2, 3]
        assert all("NaN" not in (output / name).read_text(encoding="utf-8") for name in ("run_config.json", "training_metrics.jsonl", "evaluation_metrics.jsonl", "summary.json"))
        assert result.best_checkpoint.exists() and result.final_checkpoint.exists()
        best_agent, best_metadata = load_masac_checkpoint(result.best_checkpoint)
        final_agent, final_metadata = load_masac_checkpoint(result.final_checkpoint)
        assert best_agent.num_relays == final_agent.num_relays == 1
        assert best_metadata.environment_steps in (2, 3)
        assert final_metadata.environment_steps == 3
        summary = json.loads(result.summary_file.read_text(encoding="utf-8"))
        assert summary["best_checkpoint"] == str(result.best_checkpoint)
        assert np.isfinite(result.best_mean_return)


def test_experiment_rejects_nonempty_output_and_invalid_intervals() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        output = Path(directory) / "run"
        output.mkdir()
        (output / "existing.txt").write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            run_masac_experiment(*_experiment_parts(output))
        with pytest.raises(ValueError):
            MASACExperimentConfig(Path(directory) / "other", log_interval_steps=True)
        with pytest.raises(ValueError):
            MASACExperimentConfig(Path(directory) / "other", evaluation_interval_steps=0)
