import numpy as np
import pytest

from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.policies import MPCConfig, evaluate_action_sequence, plan_mpc


def test_mpc_config_validation():
    assert MPCConfig().horizon == 3
    for kwargs in ({"horizon": 0}, {"population_size": 2}, {"iterations": 0},
                   {"elite_fraction": 0}, {"discount": 2},
                   {"minimum_standard_deviation": 0}, {"horizon": True}):
        with pytest.raises(ValueError):
            MPCConfig(**kwargs)


def test_sequence_validation_and_copy_isolation():
    env = MultiRelayEnvironment()
    env.reset(seed=4)
    before = (env.step_index, np.stack([state.position_m for state in env.states]).copy())
    valid = np.zeros((2, env.config.num_relays, 3))
    result = evaluate_action_sequence(env, valid)
    assert np.isfinite(result.discounted_return)
    assert np.isfinite(result.mean_rate_e2e_bps)
    assert env.step_index == before[0]
    assert np.array_equal(np.stack([state.position_m for state in env.states]), before[1])
    for bad in (np.zeros((env.config.num_relays, 3)), np.full(valid.shape, np.nan),
                np.full(valid.shape, 2.0)):
        with pytest.raises(ValueError):
            evaluate_action_sequence(env, bad)


@pytest.mark.parametrize("num_relays", [1, 4])
def test_mpc_shapes_reproducible_and_finite(num_relays):
    env1 = MultiRelayEnvironment()
    env1.config = type(env1.config)(**{**vars(env1.config), "num_relays": num_relays})
    env1 = MultiRelayEnvironment(env1.config)
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
