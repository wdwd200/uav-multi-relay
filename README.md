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

The following are not yet implemented: a full environment, safety-distance
filtering, reward functions, and multi-agent reinforcement learning.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Tests

```bash
pytest
```
