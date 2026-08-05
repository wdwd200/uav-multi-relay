# UAV Multi Relay

This project researches multi-relay, single-task UAV communication.

The logical topology is `H -> R1 -> ... -> RK -> L`.

## Currently Implemented

- UAV position and velocity state.
- Directional velocity and acceleration limits.
- Feasible applied-velocity calculation.
- Multi-hop geometry and air-to-air channel utility functions.
- Single-hop capacity calculation.
- Equal-time and analytical optimal TDMA.
- A synchronous, dependency-free multi-relay dynamic environment.
- Endpoint waypoint followers, relay safety filtering, observations, rewards, and an equal-spacing baseline.
- Stationary, equal-spacing, weighted-spacing, and greedy one-step coordinate-search baselines.
- Requested actions and safety-filtered executed normalized actions in environment info.
- Fixed-capacity multi-agent replay storage with deterministic sampling.
- Parameter-sharing MASAC one-batch update core with target critics and entropy temperature.
- Finite-horizon cross-entropy MPC with rolling-horizon control.

The environment accepts relay actions with shape `(K, 3)` in `[-1, 1]`.
H/L waypoint paths are generated reproducibly from `reset(seed=...)`, and the
configuration must admit a hard-feasible initial chain for its chosen `K`.
H follows a higher altitude trajectory band, while L follows a lower task
trajectory band. The initial chain is constructed from the configured distance
limits rather than accepted by random rejection sampling.

```python
import numpy as np

from uav_multi_relay import MultiRelayEnvironment

env = MultiRelayEnvironment()
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(
    np.zeros((env.config.num_relays, 3))
)
```

The environment collection and parameter-sharing MASAC training loop are implemented.
MASAC checkpoints and a deterministic independent evaluator are also available.
Replay Buffer and in-progress episode state are intentionally not included in
checkpoints, so checkpoints do not claim exact mid-episode resume.

Shared Gaussian Actor and centralized twin-Q Critic network building blocks are
implemented for the learning foundation. The replay buffer stores the
safety-filtered normalized action that was actually executed, and keeps
termination separate from time-limit truncation. Parameter-sharing MASAC now
implements action selection, critic targets, one-batch actor/critic/alpha updates,
and Polyak target updates.

The MPC planner predicts on a deep copy of the environment and optimizes the
discounted full team reward. It executes only the first action of the selected
joint sequence:

```python
from uav_multi_relay.policies import MPCConfig, mpc_actions

config = MPCConfig(horizon=2, population_size=6, iterations=2)
action = mpc_actions(env, config=config, seed=0)
```

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Tests

```bash
python -m pytest
```

## MASAC Training

Run a minimal training collection and update loop:

```bash
python scripts/train.py --steps 30 --batch-size 4 --random-action-steps 4 \
  --update-after-steps 4 --updates-per-step 1 --seed 0
```

The script infers observation dimensions from the environment reset and prints a
compact JSON summary. It does not write checkpoints or training logs.

Save a model checkpoint and evaluate it:

```bash
python scripts/train.py --steps 20 --batch-size 4 --random-action-steps 4 \
  --update-after-steps 4 --seed 0 --checkpoint-out masac_smoke.pt
python scripts/evaluate.py --checkpoint masac_smoke.pt --episodes 2 --seed 100
```

The checkpoint contains the actor, critics, target critic, entropy temperature,
optimizers, agent architecture/hyperparameters, and training counters. It does
not contain the replay buffer, current environment episode, or a full environment
configuration snapshot.
