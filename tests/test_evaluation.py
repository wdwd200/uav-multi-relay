from dataclasses import replace

import numpy as np
import torch

from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.learning import ParameterSharingMASAC
from uav_multi_relay.training import MASACEvaluationConfig, evaluate_masac


def test_evaluation_is_deterministic_and_does_not_mutate_environment_or_agent() -> None:
    base = MultiRelayEnvironment()
    env = MultiRelayEnvironment(replace(base.config, num_relays=1, max_steps=2))
    env.reset(seed=22)
    initial_observation, _ = env.reset(seed=22)
    agent = ParameterSharingMASAC(initial_observation["local"].shape[1], initial_observation["global"].shape[0], 1, hidden_dims=(8, 8), device="cpu")
    before_step = env.step_index
    before_actor = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    first = evaluate_masac(env, agent, MASACEvaluationConfig(2, 100))
    second = evaluate_masac(env, agent, MASACEvaluationConfig(2, 100))
    assert first == second
    assert env.step_index == before_step
    assert all(torch.equal(before, after) for before, after in zip(before_actor, agent.actor.parameters()))
    assert first.episodes == 2
    assert all(result.episode_length == 2 for result in first.episode_results)
    assert np.isfinite(first.mean_return)
    assert np.isfinite(first.minimum_rate_e2e_bps)


def test_evaluation_uses_configured_episode_count() -> None:
    base = MultiRelayEnvironment()
    env = MultiRelayEnvironment(replace(base.config, num_relays=4, max_steps=1))
    observation, _ = env.reset(seed=0)
    agent = ParameterSharingMASAC(observation["local"].shape[1], observation["global"].shape[0], 4, hidden_dims=(8, 8), device="cpu")
    summary = evaluate_masac(env, agent, MASACEvaluationConfig(3, 0))
    assert len(summary.episode_results) == 3
    assert summary.terminated_episode_rate == 0.0
    assert all(np.isfinite(value) for value in (summary.mean_return, summary.return_std, summary.mean_rate_e2e_bps, summary.mean_intervention_rate))
