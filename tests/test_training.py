from dataclasses import replace

import numpy as np
import pytest

from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.learning import MultiAgentReplayBuffer, ParameterSharingMASAC
from uav_multi_relay.training import MASACTrainingConfig, train_masac


class RecordingAgent:
    def __init__(self, observation: dict[str, np.ndarray]) -> None:
        self.num_relays = observation["local"].shape[0]
        self.local_observation_dim = observation["local"].shape[1]
        self.global_state_dim = observation["global"].shape[0]
        self.action_dim = 3
        self.act_calls = 0
        self.update_calls = 0

    def act(self, local: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self.act_calls += 1
        return np.zeros((self.num_relays, 3), dtype=np.float32)

    def update(self, batch: object) -> None:
        self.update_calls += 1


class RecordingEnvironment:
    def __init__(self, env: MultiRelayEnvironment) -> None:
        self.env = env
        self.config = env.config
        self.requested_actions: list[np.ndarray] = []
        self.reset_seeds: list[int | None] = []

    def reset(self, seed: int | None = None):
        self.reset_seeds.append(seed)
        return self.env.reset(seed=seed)

    def step(self, actions: object):
        self.requested_actions.append(np.asarray(actions).copy())
        return self.env.step(actions)


def _parts(num_relays: int = 2):
    base = MultiRelayEnvironment()
    env = RecordingEnvironment(MultiRelayEnvironment(replace(base.config, num_relays=num_relays)))
    observation, _ = env.reset(seed=0)
    agent = RecordingAgent(observation)
    replay = MultiAgentReplayBuffer(16, num_relays, observation["local"].shape[1], observation["global"].shape[0], 3, seed=0)
    return env, agent, replay


@pytest.mark.parametrize("kwargs", [
    {"total_environment_steps": 0}, {"replay_capacity": False}, {"batch_size": 0},
    {"random_action_steps": -1}, {"update_after_steps": -1}, {"updates_per_step": True},
    {"batch_size": 3, "replay_capacity": 2},
])
def test_training_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MASACTrainingConfig(**kwargs)


def test_random_warmup_then_actor_and_applied_actions_are_stored() -> None:
    env, agent, replay = _parts()
    summary = train_masac(env, agent, replay, MASACTrainingConfig(total_environment_steps=3, replay_capacity=16, batch_size=4, random_action_steps=2, update_after_steps=5, seed=7))
    expected = np.random.default_rng(7).uniform(-1.0, 1.0, size=(2, 3))
    assert summary.total_environment_steps == 3
    assert agent.act_calls == 1
    assert np.allclose(env.requested_actions[0], expected)
    assert np.allclose(env.requested_actions[2], 0.0)
    assert not np.allclose(replay.applied_actions[0], env.requested_actions[0])


def test_updates_wait_for_both_thresholds_and_repeat_per_step() -> None:
    env, agent, replay = _parts()
    summary = train_masac(env, agent, replay, MASACTrainingConfig(total_environment_steps=5, replay_capacity=16, batch_size=3, random_action_steps=5, update_after_steps=4, updates_per_step=2))
    assert summary.total_updates == agent.update_calls == 4
    assert replay.size == 5


def test_truncation_resets_before_the_next_step() -> None:
    base = MultiRelayEnvironment()
    env = RecordingEnvironment(MultiRelayEnvironment(replace(base.config, num_relays=1, max_steps=2)))
    observation, _ = env.reset(seed=0)
    agent = RecordingAgent(observation)
    replay = MultiAgentReplayBuffer(8, 1, observation["local"].shape[1], observation["global"].shape[0], 3)
    summary = train_masac(env, agent, replay, MASACTrainingConfig(total_environment_steps=5, replay_capacity=8, batch_size=8, random_action_steps=5, update_after_steps=9, seed=11))
    assert summary.completed_episodes == 2
    assert summary.episode_lengths == (2, 2)
    assert env.reset_seeds == [0, 11, 12, 13]
    assert len(env.requested_actions) == 5


def test_random_collection_is_reproducible_and_supports_dynamic_relay_counts() -> None:
    for relays in (1, 4):
        first = _parts(relays)
        second = _parts(relays)
        config = MASACTrainingConfig(total_environment_steps=3, replay_capacity=16, batch_size=8, random_action_steps=3, update_after_steps=9, seed=5)
        first_summary = train_masac(*first, config)
        second_summary = train_masac(*second, config)
        assert first_summary.mean_rate_e2e_bps == second_summary.mean_rate_e2e_bps
        assert np.array_equal(first[2].applied_actions[:3], second[2].applied_actions[:3])


@pytest.mark.parametrize("num_relays", [1, 4])
def test_real_masac_update_returns_finite_summary(num_relays: int) -> None:
    base = MultiRelayEnvironment()
    env = MultiRelayEnvironment(replace(base.config, num_relays=num_relays))
    observation, _ = env.reset(seed=0)
    agent = ParameterSharingMASAC(observation["local"].shape[1], observation["global"].shape[0], num_relays, hidden_dims=(8, 8))
    replay = MultiAgentReplayBuffer(8, num_relays, observation["local"].shape[1], observation["global"].shape[0], 3, seed=0)
    summary = train_masac(env, agent, replay, MASACTrainingConfig(total_environment_steps=3, replay_capacity=8, batch_size=2, random_action_steps=2, update_after_steps=2, seed=0))
    assert summary.total_updates == 2
    assert summary.last_update_metrics is not None
    assert all(np.isfinite(value) for value in vars(summary.last_update_metrics).values())
    assert np.isfinite(summary.mean_rate_e2e_bps)
    assert np.isfinite(summary.intervention_rate)
