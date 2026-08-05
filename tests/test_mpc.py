import numpy as np
import pytest
from dataclasses import replace

from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.baselines import equal_spacing_actions, stationary_actions
from uav_multi_relay.config import default_environment_config
from uav_multi_relay.policies import MPCConfig, evaluate_action_sequence, plan_mpc


def test_mpc_config_validation():
    assert MPCConfig().horizon == 3
    for kwargs in ({"horizon": 0}, {"population_size": 2}, {"iterations": 0},
                   {"elite_fraction": 0}, {"discount": 2},
                   {"minimum_standard_deviation": 0}, {"horizon": True}):
        with pytest.raises(ValueError):
            MPCConfig(**kwargs)
    for kwargs in ({"discount": np.nan}, {"discount": np.inf},
                   {"initial_standard_deviation": np.nan},
                   {"minimum_standard_deviation": 0.7}):
        with pytest.raises(ValueError):
            MPCConfig(**kwargs)


def test_sequence_validation_and_copy_isolation():
    env = MultiRelayEnvironment()
    env.reset(seed=4)
    before_step = env.step_index
    before_positions = np.stack([state.position_m for state in env.states]).copy()
    before_velocities = np.stack([state.velocity_mps for state in env.states]).copy()
    before_applied = env._last_applied_relay_velocities.copy()
    before_waypoint_indices = (env._high_follower._index, env._low_follower._index)
    valid = np.zeros((2, env.config.num_relays, 3))
    result = evaluate_action_sequence(env, valid)
    assert np.isfinite(result.discounted_return)
    assert np.isfinite(result.mean_rate_e2e_bps)
    assert env.step_index == before_step
    assert np.array_equal(np.stack([state.position_m for state in env.states]), before_positions)
    assert np.array_equal(np.stack([state.velocity_mps for state in env.states]), before_velocities)
    assert np.array_equal(env._last_applied_relay_velocities, before_applied)
    assert (env._high_follower._index, env._low_follower._index) == before_waypoint_indices
    for bad in (np.zeros((env.config.num_relays, 3)), np.full(valid.shape, np.nan),
                np.full(valid.shape, np.inf), np.full(valid.shape, 2.0),
                [["bad"]], object()):
        with pytest.raises(ValueError):
            evaluate_action_sequence(env, bad)
    with pytest.raises(ValueError):
        evaluate_action_sequence(object(), valid)


def test_truncation_and_anchor_floor_and_independent_arrays():
    base = default_environment_config()
    config = replace(base, max_steps=1)
    env = MultiRelayEnvironment(config)
    env.reset(seed=1)
    sequence = np.zeros((3, config.num_relays, 3))
    evaluation = evaluate_action_sequence(env, sequence)
    assert evaluation.steps_evaluated == 1
    assert evaluation.truncated
    small = MPCConfig(horizon=2, population_size=6, iterations=2, elite_fraction=0.5)
    plan = plan_mpc(env, small, seed=3)
    zero = evaluate_action_sequence(env, np.repeat(stationary_actions(env)[None], 2, axis=0), small.discount)
    equal = evaluate_action_sequence(env, np.repeat(equal_spacing_actions(env)[None], 2, axis=0), small.discount)
    assert plan.predicted_return >= max(zero.discounted_return, equal.discounted_return)
    sequence_value = plan.action_sequence[0, 0, 0]
    plan.first_action[0, 0] = -0.123
    assert plan.action_sequence[0, 0, 0] == sequence_value


@pytest.mark.parametrize("num_relays", [1, 4])
def test_mpc_shapes_reproducible_and_finite(num_relays):
    base = default_environment_config()
    env_config = replace(base, num_relays=num_relays)
    env1 = MultiRelayEnvironment(env_config)
    env1.reset(seed=2)
    env2 = MultiRelayEnvironment(env1.config)
    env2.reset(seed=2)
    config = MPCConfig(horizon=2, population_size=6, iterations=2, elite_fraction=0.5)
    plan1 = plan_mpc(env1, config, seed=7)
    plan2 = plan_mpc(env2, config, seed=7)
    assert plan1.first_action.shape == (num_relays, 3)
    assert plan1.action_sequence.shape == (2, num_relays, 3)
    assert np.all(np.isfinite(plan1.action_sequence))
    assert np.all((plan1.action_sequence >= -1) & (plan1.action_sequence <= 1))
    assert np.array_equal(plan1.action_sequence, plan2.action_sequence)


def test_mpc_can_roll_for_ten_real_steps():
    env = MultiRelayEnvironment()
    env.reset(seed=8)
    config = MPCConfig(horizon=2, population_size=6, iterations=1, elite_fraction=0.5)
    for _ in range(10):
        if env.step_index >= env.config.max_steps:
            break
        plan = plan_mpc(env, config, seed=env.step_index)
        _, reward, terminated, truncated, _ = env.step(plan.first_action)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
