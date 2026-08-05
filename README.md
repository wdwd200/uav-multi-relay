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

Multi-agent reinforcement learning remains intentionally unimplemented.
Model-predictive control is also not implemented.

Shared Gaussian Actor and centralized twin-Q Critic network building blocks are
implemented for the learning foundation. The replay buffer stores the
safety-filtered normalized action that was actually executed, and keeps
termination separate from time-limit truncation. Parameter-sharing MASAC now
implements action selection, critic targets, one-batch actor/critic/alpha updates,
and Polyak target updates. Environment collection loops, checkpoints, and full
training experiments are not implemented.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Tests

```bash
python -m pytest
```
