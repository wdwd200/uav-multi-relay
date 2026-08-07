import copy
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.learning import MAPPOAgent, MAPPOConfig, MAPPORollout
from uav_multi_relay.training import (
    MAPPOCheckpointMetadata,
    MAPPOExperimentConfig,
    MAPPOTrainingConfig,
    load_mappo_checkpoint,
    run_mappo_experiment,
    save_mappo_checkpoint,
)


def _parts(directory: Path):
    base = MultiRelayEnvironment()
    config = replace(base.config, num_relays=1, max_steps=2)
    train = MultiRelayEnvironment(config)
    evaluate = MultiRelayEnvironment(config)
    observation, _ = train.reset(seed=0)
    agent = MAPPOAgent(observation["local"].shape[1], observation["global"].shape[0], 1, hidden_dims=(8, 8), config=MAPPOConfig(update_epochs=1, mini_batch_size=2))
    return train, evaluate, agent, MAPPOTrainingConfig(4, 2, 0), MAPPOExperimentConfig(directory, 2, 2, 1, 100, 2)


def _optimizer_state_equal(left: dict, right: dict) -> bool:
    if left["param_groups"] != right["param_groups"] or left["state"].keys() != right["state"].keys():
        return False
    for key in left["state"]:
        for name, value in left["state"][key].items():
            other = right["state"][key][name]
            if isinstance(value, torch.Tensor):
                if not torch.equal(value, other):
                    return False
            elif value != other:
                return False
    return True


def test_experiment_writes_artifacts_and_partial_rollout_summary() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        output = Path(directory) / "run"
        result = run_mappo_experiment(*_parts(output))
        assert {item.name for item in output.iterdir()} == {"run_config.json", "training_metrics.jsonl", "evaluation_metrics.jsonl", "summary.json", "best_checkpoint.pt", "final_checkpoint.pt", "checkpoints"}
        summary = json.loads(result.summary_file.read_text(encoding="utf-8"))
        assert summary["total_updates"] == 2
        assert summary["discarded_partial_rollout_steps"] == 0


def test_checkpoint_round_trip_compares_original_and_loaded_agent() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        _, _, agent, _, _ = _parts(Path(directory) / "unused")
        observation = np.zeros((agent.num_relays, agent.local_observation_dim), dtype=np.float32)
        global_state = np.zeros(agent.global_state_dim, dtype=np.float32)
        before_action = agent.act(observation, deterministic=True)
        before_value = agent.value(global_state)
        actor_parameters = [item.detach().clone() for item in agent.actor.parameters()]
        critic_parameters = [item.detach().clone() for item in agent.value_critic.parameters()]
        actor_optimizer = copy.deepcopy(agent.actor_optimizer.state_dict())
        critic_optimizer = copy.deepcopy(agent.critic_optimizer.state_dict())
        metadata = MAPPOCheckpointMetadata(2, 1, 1)
        path = save_mappo_checkpoint(Path(directory) / "agent.pt", agent, metadata)
        loaded, loaded_metadata = load_mappo_checkpoint(path)
        assert np.array_equal(before_action, loaded.act(observation, deterministic=True))
        assert before_value == pytest.approx(loaded.value(global_state))
        assert all(torch.equal(left, right) for left, right in zip(actor_parameters, loaded.actor.parameters()))
        assert all(torch.equal(left, right) for left, right in zip(critic_parameters, loaded.value_critic.parameters()))
        assert _optimizer_state_equal(actor_optimizer, loaded.actor_optimizer.state_dict())
        assert _optimizer_state_equal(critic_optimizer, loaded.critic_optimizer.state_dict())
        assert loaded.config == agent.config
        assert loaded_metadata == metadata


def test_checkpoint_rejects_corruption_and_nonempty_output() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        path = Path(directory) / "bad.pt"
        path.write_bytes(b"bad")
        with pytest.raises(ValueError):
            load_mappo_checkpoint(path)
        output = Path(directory) / "run"
        output.mkdir()
        (output / "x").write_text("x")
        with pytest.raises(ValueError):
            run_mappo_experiment(*_parts(output))


def test_checkpoint_restores_agent_that_can_continue_update() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        _, _, agent, _, _ = _parts(Path(directory) / "unused")
        path = save_mappo_checkpoint(Path(directory) / "agent.pt", agent, MAPPOCheckpointMetadata(2, 1, 1))
        loaded, _ = load_mappo_checkpoint(path)
        rollout = MAPPORollout(2, 1, agent.local_observation_dim, agent.global_state_dim)
        for _ in range(2):
            rollout.add(np.zeros((1, agent.local_observation_dim), np.float32), np.zeros(agent.global_state_dim, np.float32), np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32), np.zeros((1, 1), np.float32), 1.0, 0.0, 0.0, False, False)
        assert np.isfinite(loaded.update(rollout).policy_loss)
