# Stage 4D: ten-policy comparison and communication-model sensitivity

## Outcome

Stage 4D is complete. MATD3 and MADDPG each trained from scratch for 20,000
environment steps under the prescribed configuration, then all four learning
algorithms and six baselines were evaluated on the same ten episode seeds.
This result is **classification C**: every learning algorithm remains clearly
below the rule baselines. This is a performance result, not a program error;
no 4D hyperparameter changes were made.

## Reproducibility and implementation changes

- The deterministic CLI now passes `seed=args.seed` to `MultiAgentReplayBuffer`.
- The deterministic trainer now names the per-step quantity
  `termination_event_rate_per_step` accurately and additionally records
  `terminated_episode_rate`, `mean_episode_length`, and `mean_episode_return`.
- A same-seed end-to-end MATD3 run (`seed=17`) produced byte-identical training
  and evaluation JSONL, equal core summary values, and equal final Actor
  parameters. A `seed=18` run produced different Actor parameters.
- Environment configuration now supports backward-compatible `tdma_mode`
  (`optimal` or `equal`) and `antenna_mode` (`dipole` or `isotropic`). Defaults
  remain optimal TDMA and dipole gain; an explicit-default regression test
  confirms the same rate as before.

All deterministic Critic updates continue to use replayed safety-filtered
**applied** actions. Actors and target actors emit requested actions. Because
the safety filter is non-differentiable, this remains an approximate off-policy
action-semantics treatment shared by MASAC, MATD3, and MADDPG.

## Verification

```text
python -m pytest
203 passed in 67.78s

python -m compileall -q src tests scripts
success
```

The three new tests cover default communication-mode equivalence, valid equal
TDMA/isotropic behavior, and full seeded deterministic training reproducibility.
Existing MATD3/MADDPG tests retain target masks, smoothing, delay, applied
action, checkpoint, and checkpoint-type coverage.

## Formal training

Both runs used the specified 90 m waypoint radius, 250 maximum steps, batch
size 256, 2,000 random/update warm-up steps, `updates_per_step=1`, exploration
noise 0.1, seed 0, CPU, and reward weights `(1,1,1,0.1,0.1,1)`.

| Algorithm | Steps / updates | Best eval return | Training terminated-episode rate | Mean episode length | Intervention / mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| MATD3 | 20,000 / 18,001 | 177.4902 | 0.9959 | 41.37 | 0.9991 / 1.0000 |
| MADDPG | 20,000 / 18,001 | 161.3195 | 0.9963 | 37.04 | 0.9970 / 1.0000 |

Each output contains `step_000000.pt` through `step_020000.pt` at the required
2,500-step interval, plus best and final checkpoints, 20 training log records,
8 evaluation records, finite metrics, and a JSON summary. These outputs are
ignored by Git.

## Ten-policy comparison

All strategies used seeds 20000–20009, max 250 steps, 90 m waypoint radius,
optimal TDMA, dipole antenna, and the required weights. Values below are means
over ten episodes; `rate` is Mbps.

| Policy | Return ± std | Return/step | Rate | Terminated eps. | Length | Intervention | Mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MAPPO | 921.29 ± 174.23 | 4.143 | 41.823 | 0.4 | 222.6 | 0.0088 | 0.0088 |
| MASAC | 341.27 ± 162.67 | 3.955 | 41.058 | 1.0 | 85.8 | 1.0 | 1.0 |
| MATD3 | 226.45 ± 68.18 | 3.941 | 41.317 | 1.0 | 57.7 | 1.0 | 1.0 |
| MADDPG | 141.50 ± 38.09 | 3.965 | 42.140 | 1.0 | 35.6 | 1.0 | 1.0 |
| Random | 567.94 ± 287.08 | 4.039 | 42.093 | 0.8 | 139.6 | 1.0 | 1.0 |
| Stationary | 1068.00 ± 31.64 | 4.272 | 42.955 | 0.0 | 250.0 | 0.0 | 0.0 |
| Equal spacing | 1073.70 ± 39.20 | 4.295 | 43.752 | 0.0 | 250.0 | 0.9952 | 0.9952 |
| Weighted spacing | 1073.70 ± 39.20 | 4.295 | 43.752 | 0.0 | 250.0 | 0.9952 | 0.9952 |
| Greedy | 57.67 ± 1.37 | 3.870 | 43.774 | 1.0 | 14.9 | 1.0 | 1.0 |
| MPC | 1068.00 ± 31.64 | 4.272 | 42.955 | 0.0 | 250.0 | 0.0 | 0.0 |

The 100 episode records and all summary fields, including minimum rate and mean
action-computation time, are retained in the ignored comparison output.

## Frozen-policy communication-model sensitivity

This is a frozen-policy sensitivity experiment, not a retraining ablation.
The same six policies and seeds were evaluated after only changing the stated
communication model.

| Scenario | MAPPO | MASAC | MATD3 | MADDPG | Stationary | Equal spacing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: optimal TDMA + dipole | 921.29 | 341.27 | 226.45 | 141.50 | 1068.00 | 1073.70 |
| B: equal TDMA + dipole | 797.05 | 317.20 | 207.38 | 129.48 | 986.74 | 1065.60 |
| C: optimal TDMA + isotropic | 854.60 | 312.04 | 206.69 | 128.89 | 970.00 | 968.70 |

Changing either communication component reduces frozen-policy returns in this
evaluation. It does not show how policies would perform if retrained under the
changed model.

## Git and next stage

- Code commit: `455bb59` — `fix: make deterministic MARL experiments reproducible`
- Code push: successful to `origin/main`.
- Result report commit: recorded by its containing documentation commit.
- No environment termination logic, MASAC algorithm, or MAPPO algorithm was
  modified; no failures were filtered; no `outputs/` artifact is tracked.

Next task: **Stage 4E — core-structure and dynamic-scenario ablations.**
