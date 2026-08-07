# Stage 4C: MATD3 and MADDPG implementation result

## Scope and outcome

Stage 4C is complete. It adds parameter-sharing MATD3 and MADDPG without
changing the environment, reward calculation, safety filter, MASAC, MAPPO, or
rule-based baselines. This stage implements and smoke-tests the algorithms; it
does not make a performance claim. The next stage is 4D formal training,
ten-policy comparison, and controlled ablation.

## Delivered implementation

- `SharedDeterministicActor` is a parameter-sharing, tanh-bounded Actor whose
  output has shape `(batch, K, action_dim)` and lies in `[-1, 1]`.
- MADDPG uses a centralized single Critic, Actor/Critic targets, a true-
  termination bootstrap mask, and Polyak updates every update.
- MATD3 uses centralized twin Critics, clipped target-policy Gaussian noise,
  minimum target Q, and `policy_delay=2`; Actor and all target networks update
  only on delayed steps.
- Both algorithms share replay collection, training, deterministic evaluation,
  checkpointing, experiment artifacts, and thin CLI entry points.
- Checkpoints record the algorithm type, dimensions, all online/target networks,
  optimizers, configuration, and environment/update/episode counters. Loading a
  checkpoint as the wrong algorithm is rejected.
- The comparison CLI accepts `--matd3-checkpoint` and `--maddpg-checkpoint`;
  one comparison can now evaluate MAPPO, MASAC, MATD3, MADDPG, Random,
  Stationary, Equal Spacing, Weighted Spacing, Greedy, and MPC on identical
  episode seeds. Intervention rate and requested/applied mismatch rate remain
  separate fields.

## Action and replay semantics

The Actor emits a requested normalized action. The environment safety filter
produces the applied normalized action, and the existing `MultiAgentReplayBuffer`
continues to save that applied action. Critic transition updates use replay
`applied_actions`; Actor updates and TD3 target actions use direct Actor
requested actions.

The safety filter is non-differentiable. Consequently the Actor update is not
an exact constrained policy gradient; this off-policy action-semantics
limitation is shared by the present MASAC, MATD3, and MADDPG implementations.
Requested/applied mismatch is diagnostic only and was not used to alter safety
behavior.

The prior 4B comparison establishes only that the then-current MAPPO
implementation outperformed the then-current MASAC implementation under that
configuration. It does not by itself identify action semantics as the cause.

## Verification

Commands run successfully:

```text
python -m pytest
200 passed in 67.99s

python -m compileall -q src tests scripts
success

python -m pytest -q tests/test_deterministic_learning.py \
  tests/test_deterministic_training.py tests/test_deterministic_experiment.py
6 passed in 4.37s
```

The full suite retains the 194 previous tests and adds 6 deterministic-algorithm
tests. They cover Actor/Critic shape, range, finite values and repeatability;
MADDPG termination masking and update behavior; MATD3 delayed Actor updates,
clipped smoothing noise, and replay-action Critic use; dynamic relay counts;
training collection; checkpoint round-trip; wrong-type checkpoint rejection;
and continued deterministic action equivalence after loading.

## Smoke experiments

Both required CPU smoke commands used the Stage 4C fixed 1,000-step settings
with `max_steps=50`, waypoint radius 90 m, batch size 64, 200 random/update
warm-up steps, evaluation/checkpoint interval 500, seed 0, and evaluation seed
10000. Each completed 1,000 environment steps, made updates, emitted finite
JSON/JSONL, and produced `step_000000.pt`, periodic checkpoints,
`best_checkpoint.pt`, and `final_checkpoint.pt`.

| Algorithm | Output directory | Best deterministic evaluation mean return |
| --- | --- | ---: |
| MATD3 | `outputs/stage4c_matd3_smoke` | 93.79100114932785 |
| MADDPG | `outputs/stage4c_maddpg_smoke` | 74.68084970121981 |

These values are smoke-only and are not a performance comparison.

A one-episode integration smoke then loaded the existing MAPPO/MASAC final
checkpoints alongside both new smoke checkpoints and evaluated all four
learning policies in a single comparison invocation. It completed successfully
and confirmed the shared comparison path and common seeds; the generated output
is ignored by Git.

## Git and next task

- Code commit: `1c9bbfb` (`feat: implement parameter-sharing MATD3 and MADDPG`)
- Code push: successful to `origin/main`.
- No run output, checkpoint, JSONL artifact, cache, or temporary log is tracked.

Next recommended task: **Stage 4D — deterministic-algorithm formal training,
ten-policy comparison, and core ablations.**
