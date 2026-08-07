from dataclasses import replace
import numpy as np
from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.learning import MAPPOAgent, MAPPOConfig
from uav_multi_relay.training import MAPPOTrainingConfig, train_mappo


def _parts():
    base = MultiRelayEnvironment()
    env = MultiRelayEnvironment(replace(base.config, num_relays=1, max_steps=2))
    observation, _ = env.reset(seed=0)
    agent = MAPPOAgent(observation["local"].shape[1], observation["global"].shape[0], 1, hidden_dims=(8, 8), config=MAPPOConfig(update_epochs=1, mini_batch_size=2))
    return env, agent


def test_training_collects_full_rollouts_and_updates() -> None:
    env, agent = _parts()
    progress = []
    summary = train_mappo(env, agent, MAPPOTrainingConfig(4, 2, 3), progress_interval_steps=2, progress_callback=progress.append)
    assert summary.total_updates == 2
    assert summary.completed_episodes == 2
    assert summary.discarded_partial_rollout_steps == 0
    assert [item.environment_steps for item in progress] == [2, 4]
    assert summary.last_update_metrics is not None
    assert np.isfinite(summary.mean_rate_e2e_bps)


def test_training_records_discarded_partial_rollout_without_updating_it() -> None:
    env, agent = _parts()
    summary = train_mappo(env, agent, MAPPOTrainingConfig(3, 2, 0))
    assert summary.total_updates == 1
    assert summary.discarded_partial_rollout_steps == 1
