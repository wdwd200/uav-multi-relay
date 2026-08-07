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

## MAPPO Training

Stage 4A adds an independent parameter-sharing MAPPO implementation. Its PPO
ratio is defined over the Actor's sampled **requested normalized actions**;
the environment's safety-filtered applied actions are retained only for
diagnostics. MAPPO uses a centralized value critic, fixed on-policy rollouts,
GAE with distinct terminated/truncated masks, atomic MAPPO-only checkpoints,
and deterministic evaluation. Run a short experiment with
`python scripts/run_mappo_experiment.py --help`.

## MASAC Training

Run a minimal training collection and update loop:

```bash
python scripts/train.py --steps 30 --batch-size 4 --random-action-steps 4 \
  --update-after-steps 4 --updates-per-step 1 --seed 0
```

The script infers observation dimensions from the environment reset and prints a
compact JSON summary. By default it writes no training logs; `--checkpoint-out`
optionally saves one final checkpoint. Complete logs, periodic evaluation, and
best-checkpoint selection are handled by `run_experiment.py`.

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

## Reward and Scenario Diagnostics

`EnvironmentConfig.reward_weights` exposes explicit non-negative coefficients
for rate, link, separation, intervention, motion, and failure terms. Defaults
are all `1.0`, preserving the original reward scale. Motion cost measures only
controlled relay velocities and accelerations; endpoint H/L motion is excluded.
`info["reward_terms"]` retains raw components and includes `weighted_reward`.

Generate a short scenario matrix without training MASAC:

```bash
python scripts/diagnose_scenarios.py --output scenario_diagnostics.json \
  --radii 30 60 90 120 --max-steps 100 250 --episodes 5 \
  --seed 30000 --policies stationary equal_spacing --num-relays 4
```

The JSON contains per-episode failure reasons and summaries for displacement,
relay path length, link capacity/distance, reward components, termination, and
intervention rates. Existing output files are never overwritten. Diagnostics
are scenario evidence, not formal training or multi-seed experimental claims.

## MASAC Experiments

Run one logged training experiment with periodic deterministic evaluation:

```bash
python scripts/run_experiment.py --output-dir masac_experiment_smoke \
  --steps 20 --batch-size 4 --random-action-steps 4 \
  --update-after-steps 4 --log-interval 5 --evaluation-interval 10 \
  --evaluation-episodes 2 --seed 0 --evaluation-seed 100
```

Each run directory contains `run_config.json`, training and evaluation JSONL
logs, `best_checkpoint.pt`, `final_checkpoint.pt`, and `summary.json`. Existing
non-empty directories are rejected. Periodic evaluations reuse fixed seeds for
longitudinal comparison. Batch multi-seed experiments are not implemented.

For a stability/action diagnostic run, add `--diagnostics --checkpoint-interval 2500`.
This additionally writes interval action statistics, bounded per-failure traces,
and `checkpoints/step_*.pt`. Diagnose the completed run without changing its
training trajectory:

```bash
python scripts/diagnose_masac.py --run-dir masac_experiment \
  --output-dir masac_experiment/diagnostics --evaluation-episodes 5 \
  --evaluation-seed 10000 --comparison-episodes 10 --comparison-seed 20000
```

The diagnostic directory contains checkpoint evolution, policy diagnostics,
reward contributions, failure summaries, and a Markdown/JSON summary. These are
run artifacts and are not committed. The current repository execution report is
named for its stage and task; see
[`STAGE_4A_R1_MAPPO_SEMANTICS_REPAIR_REPORT.md`](STAGE_4A_R1_MAPPO_SEMANTICS_REPAIR_REPORT.md).

## Policy Comparison

Compare a saved MASAC checkpoint with the random, stationary, equal-spacing,
greedy, and MPC baselines on the same seeded episodes:

```bash
python scripts/compare_baselines.py --checkpoint best_checkpoint.pt \
  --output-dir comparison --episodes 3 --seed 20000 --max-steps 100 \
  --policies masac random stationary equal_spacing greedy mpc \
  --greedy-sweeps 1 --mpc-horizon 2 --mpc-population-size 8 \
  --mpc-iterations 2
```

The output directory contains the resolved comparison configuration, one JSONL
record per policy episode, and per-policy summary metrics. Every policy receives
the same episode seeds. These single-seed development comparisons validate the
workflow only; they are not formal multi-seed experimental conclusions.
