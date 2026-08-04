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

The environment accepts relay actions with shape `(K, 3)` in `[-1, 1]`.
H/L waypoint paths are generated reproducibly from `reset(seed=...)`, and the
configuration must admit a hard-feasible initial chain for its chosen `K`.

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

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Tests

```bash
python -m pytest
```
