"""Run one logged MASAC training and periodic deterministic evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.learning import MultiAgentReplayBuffer, ParameterSharingMASAC
from uav_multi_relay.training import MASACExperimentConfig, MASACTrainingConfig, run_masac_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-relays", type=int, default=4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--random-action-steps", type=int, default=1_000)
    parser.add_argument("--update-after-steps", type=int, default=1_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--evaluation-interval", type=int, default=5_000)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    base = MultiRelayEnvironment()
    environment_config = replace(
        base.config,
        num_relays=args.num_relays,
        **({"max_steps": args.max_steps} if args.max_steps is not None else {}),
    )
    training_env = MultiRelayEnvironment(environment_config)
    evaluation_env = MultiRelayEnvironment(environment_config)
    observation, _ = training_env.reset(seed=args.seed)
    local_dim = int(observation["local"].shape[-1])
    global_dim = int(observation["global"].shape[-1])
    replay = MultiAgentReplayBuffer(100_000 if args.batch_size <= 100_000 else args.batch_size, args.num_relays, local_dim, global_dim, 3, seed=args.seed)
    agent = ParameterSharingMASAC(local_dim, global_dim, args.num_relays, action_dim=3, device=args.device)
    training_config = MASACTrainingConfig(args.steps, replay.capacity, args.batch_size, args.random_action_steps, args.update_after_steps, args.updates_per_step, args.seed)
    experiment_config = MASACExperimentConfig(args.output_dir, args.log_interval, args.evaluation_interval, args.evaluation_episodes, args.evaluation_seed)
    result = run_masac_experiment(training_env, evaluation_env, agent, replay, training_config, experiment_config)
    print(json.dumps({
        "output_directory": str(result.output_directory),
        "final_checkpoint": str(result.final_checkpoint),
        "best_checkpoint": str(result.best_checkpoint),
        "summary_file": str(result.summary_file),
        "best_mean_return": result.best_mean_return,
    }, allow_nan=False))


if __name__ == "__main__":
    main()
