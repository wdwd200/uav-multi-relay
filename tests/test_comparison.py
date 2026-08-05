import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.analysis import PolicyComparisonConfig, compare_policies
from uav_multi_relay.learning import ParameterSharingMASAC
from uav_multi_relay.policies import MPCConfig
from uav_multi_relay.training import MASACCheckpointMetadata, save_masac_checkpoint


def _parts(relays: int = 1, max_steps: int = 1):
    base = MultiRelayEnvironment()
    env = MultiRelayEnvironment(replace(base.config, num_relays=relays, max_steps=max_steps))
    observation, _ = env.reset(seed=0)
    agent = ParameterSharingMASAC(observation["local"].shape[1], observation["global"].shape[0], relays, hidden_dims=(8, 8), device="cpu")
    return env, agent


def test_comparison_config_rejects_invalid_and_duplicate_policies() -> None:
    with pytest.raises(ValueError):
        PolicyComparisonConfig(0)
    with pytest.raises(ValueError):
        PolicyComparisonConfig(policies=("masac", "masac"))
    with pytest.raises(ValueError):
        PolicyComparisonConfig(policies=("unknown",))
    with pytest.raises(ValueError):
        PolicyComparisonConfig(greedy_sweeps=-1)


def test_all_policies_use_same_episode_seeds_and_finite_metrics() -> None:
    env, agent = _parts()
    before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    config = PolicyComparisonConfig(
        episodes=2,
        seed=20,
        policies=("masac", "random", "stationary", "equal_spacing", "greedy", "mpc"),
        greedy_sweeps=0,
        mpc_config=MPCConfig(horizon=1, population_size=3, iterations=1, elite_fraction=0.5),
    )
    result = compare_policies(env, agent, config)
    assert [item.episode_seed for item in result.episode_results if item.policy == "masac"] == [20, 21]
    assert all(item.episode_length == 1 for item in result.episode_results)
    assert all(np.isfinite(value) for summary in result.policy_summaries for value in vars(summary).values() if isinstance(value, (int, float)))
    assert all(item.mean_action_compute_time_s >= 0.0 for item in result.episode_results)
    assert all(torch.equal(before, after) for before, after in zip(before, agent.actor.parameters()))
    assert env.step_index == 0


def test_random_policy_is_reproducible() -> None:
    first_env, first_agent = _parts(max_steps=2)
    second_env, second_agent = _parts(max_steps=2)
    config = PolicyComparisonConfig(2, 50, ("random",))
    first = compare_policies(first_env, first_agent, config)
    second = compare_policies(second_env, second_agent, config)
    assert [item.episode_return for item in first.episode_results] == [item.episode_return for item in second.episode_results]


@pytest.mark.parametrize("relays", [1, 4])
def test_comparison_supports_dynamic_relay_counts(relays: int) -> None:
    env, agent = _parts(relays)
    result = compare_policies(env, agent, PolicyComparisonConfig(1, 10, ("masac", "stationary"), 0, MPCConfig(horizon=1, population_size=3, iterations=1, elite_fraction=0.5)))
    assert {summary.policy for summary in result.policy_summaries} == {"masac", "stationary"}


def test_comparison_script_writes_finite_json_and_rejects_nonempty_output() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        env, agent = _parts()
        checkpoint = save_masac_checkpoint(root / "agent.pt", agent, MASACCheckpointMetadata(0, 0, 0))
        output = root / "comparison"
        command = [sys.executable, "scripts/compare_baselines.py", "--checkpoint", str(checkpoint), "--output-dir", str(output), "--episodes", "1", "--seed", "1", "--max-steps", "1", "--policies", "masac", "random", "--device", "cpu"]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        assert json.loads(completed.stdout)["policies"] == ["masac", "random"]
        assert {path.name for path in output.iterdir()} == {"comparison_config.json", "comparison_episodes.jsonl", "comparison_summary.json"}
        assert all("NaN" not in path.read_text(encoding="utf-8") for path in output.iterdir())
        rejected = subprocess.run(command, capture_output=True, text=True)
        assert rejected.returncode != 0
