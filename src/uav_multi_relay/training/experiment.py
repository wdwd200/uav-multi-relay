"""Single-run MASAC experiment orchestration and artifact writing."""

from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..environment import MultiRelayEnvironment
from ..learning import MultiAgentReplayBuffer, ParameterSharingMASAC
from .checkpoints import MASACCheckpointMetadata, save_masac_checkpoint
from .evaluator import MASACEvaluationConfig, MASACEvaluationSummary, evaluate_masac
from .trainer import MASACTrainingConfig, MASACTrainingProgress, MASACTrainingSummary, _observation_arrays, train_masac


@dataclass(frozen=True)
class MASACExperimentConfig:
    output_directory: str | Path
    log_interval_steps: int = 1_000
    evaluation_interval_steps: int = 5_000
    evaluation_episodes: int = 10
    evaluation_seed: int = 10_000

    def __post_init__(self) -> None:
        output = Path(self.output_directory)
        if output.exists() and not output.is_dir():
            raise ValueError("output_directory must be a directory")
        object.__setattr__(self, "output_directory", output)
        for name in ("log_interval_steps", "evaluation_interval_steps", "evaluation_episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.evaluation_seed, bool) or not isinstance(self.evaluation_seed, Integral):
            raise ValueError("evaluation_seed must be an integer")


@dataclass(frozen=True)
class MASACExperimentResult:
    output_directory: Path
    final_checkpoint: Path
    best_checkpoint: Path
    training_log: Path
    evaluation_log: Path
    summary_file: Path
    best_mean_return: float


def _metrics_payload(metrics: object) -> dict[str, float | None]:
    names = ("critic_loss", "actor_loss", "alpha_loss", "alpha")
    if metrics is None:
        return {name: None for name in names}
    values = {name: float(getattr(metrics, name)) for name in names}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("training update metrics must be finite")
    return values


def _write_json_line(handle: Any, payload: dict[str, object]) -> None:
    handle.write(json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")
    handle.flush()


def _evaluation_payload(environment_steps: int, result: MASACEvaluationSummary) -> dict[str, object]:
    return {
        "environment_steps": environment_steps,
        "mean_return": result.mean_return,
        "return_std": result.return_std,
        "mean_rate_e2e_bps": result.mean_rate_e2e_bps,
        "minimum_rate_e2e_bps": result.minimum_rate_e2e_bps,
        "mean_intervention_rate": result.mean_intervention_rate,
        "terminated_episode_rate": result.terminated_episode_rate,
    }


def run_masac_experiment(
    training_env: MultiRelayEnvironment,
    evaluation_env: MultiRelayEnvironment,
    agent: ParameterSharingMASAC,
    replay_buffer: MultiAgentReplayBuffer,
    training_config: MASACTrainingConfig,
    experiment_config: MASACExperimentConfig,
) -> MASACExperimentResult:
    if not isinstance(experiment_config, MASACExperimentConfig):
        raise ValueError("experiment_config must be a MASACExperimentConfig")
    output = experiment_config.output_directory
    if output.exists() and any(output.iterdir()):
        raise ValueError("output_directory must be missing or empty")
    output.mkdir(parents=True, exist_ok=True)
    training_observation, _ = training_env.reset(seed=training_config.seed)
    local, global_state = _observation_arrays(training_observation)
    if local.shape != (agent.num_relays, agent.local_observation_dim) or global_state.shape != (agent.global_state_dim,):
        raise ValueError("agent and training environment observations are incompatible")
    if evaluation_env.config.num_relays != agent.num_relays:
        raise ValueError("evaluation environment relay count is incompatible")

    run_config_path = output / "run_config.json"
    training_log_path = output / "training_metrics.jsonl"
    evaluation_log_path = output / "evaluation_metrics.jsonl"
    best_checkpoint_path = output / "best_checkpoint.pt"
    final_checkpoint_path = output / "final_checkpoint.pt"
    summary_path = output / "summary.json"
    run_config = {
        "training_config": asdict(training_config),
        "experiment_config": {**asdict(experiment_config), "output_directory": str(output)},
        "agent_config": {
            name: (list(getattr(agent, name)) if name == "hidden_dims" else getattr(agent, name))
            for name in ("local_observation_dim", "global_state_dim", "num_relays", "action_dim", "hidden_dims", "gamma", "tau", "actor_learning_rate", "critic_learning_rate", "alpha_learning_rate", "initial_alpha", "target_entropy")
        },
        "observation_dimensions": {"local": list(local.shape), "global": list(global_state.shape)},
        "action_dimension": agent.action_dim,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    with run_config_path.open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, allow_nan=False, indent=2)

    best_mean_return = -float("inf")
    latest_evaluation: MASACEvaluationSummary | None = None
    with training_log_path.open("w", encoding="utf-8") as training_handle, evaluation_log_path.open("w", encoding="utf-8") as evaluation_handle:
        def progress_callback(progress: MASACTrainingProgress) -> None:
            nonlocal best_mean_return, latest_evaluation
            should_log = (
                progress.environment_steps % experiment_config.log_interval_steps == 0
                or progress.environment_steps == training_config.total_environment_steps
            )
            should_evaluate = (
                progress.environment_steps % experiment_config.evaluation_interval_steps == 0
                or progress.environment_steps == training_config.total_environment_steps
            )
            if should_log:
                payload = {
                    "environment_steps": progress.environment_steps,
                    "total_updates": progress.total_updates,
                    "completed_episodes": progress.completed_episodes,
                    "replay_size": progress.replay_size,
                    "mean_rate_e2e_bps": progress.mean_rate_e2e_bps,
                    "intervention_rate": progress.intervention_rate,
                    **_metrics_payload(progress.last_update_metrics),
                }
                _write_json_line(training_handle, payload)
            if not should_evaluate:
                return
            latest_evaluation = evaluate_masac(
                evaluation_env,
                agent,
                MASACEvaluationConfig(experiment_config.evaluation_episodes, experiment_config.evaluation_seed),
            )
            _write_json_line(evaluation_handle, _evaluation_payload(progress.environment_steps, latest_evaluation))
            if latest_evaluation.mean_return > best_mean_return:
                best_mean_return = latest_evaluation.mean_return
                save_masac_checkpoint(best_checkpoint_path, agent, MASACCheckpointMetadata(progress.environment_steps, progress.total_updates, progress.completed_episodes))

        training_summary = train_masac(
            training_env,
            agent,
            replay_buffer,
            training_config,
            progress_interval_steps=math.gcd(
                experiment_config.log_interval_steps,
                experiment_config.evaluation_interval_steps,
            ),
            progress_callback=progress_callback,
        )
    save_masac_checkpoint(final_checkpoint_path, agent, MASACCheckpointMetadata(training_summary.total_environment_steps, training_summary.total_updates, training_summary.completed_episodes))
    if latest_evaluation is None or not best_checkpoint_path.is_file():
        raise ValueError("final evaluation did not produce a best checkpoint")
    summary_payload = {
        "total_environment_steps": training_summary.total_environment_steps,
        "total_updates": training_summary.total_updates,
        "completed_episodes": training_summary.completed_episodes,
        "training_mean_rate_e2e_bps": training_summary.mean_rate_e2e_bps,
        "training_intervention_rate": training_summary.intervention_rate,
        "final_evaluation": _evaluation_payload(training_summary.total_environment_steps, latest_evaluation),
        "best_mean_return": best_mean_return,
        "best_checkpoint": str(best_checkpoint_path),
        "final_checkpoint": str(final_checkpoint_path),
        "training_log": str(training_log_path),
        "evaluation_log": str(evaluation_log_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, allow_nan=False, indent=2)
    return MASACExperimentResult(output, final_checkpoint_path, best_checkpoint_path, training_log_path, evaluation_log_path, summary_path, float(best_mean_return))


__all__ = ["MASACExperimentConfig", "MASACExperimentResult", "run_masac_experiment"]
