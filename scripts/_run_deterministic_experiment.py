"""Shared CLI implementation for reproducible deterministic MARL experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uav_multi_relay import MultiRelayEnvironment, RewardWeights, scenario_environment_config
from uav_multi_relay.learning import MultiAgentReplayBuffer, ParameterSharingMADDPG, ParameterSharingMATD3
from uav_multi_relay.training import DeterministicExperimentConfig, DeterministicTrainingConfig, run_deterministic_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--waypoint-radius", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--random-action-steps", type=int, default=2_000)
    parser.add_argument("--update-after-steps", type=int, default=2_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--exploration-noise-std", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--evaluation-interval", type=int, default=5_000)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reward-rate", type=float, default=1.0)
    parser.add_argument("--reward-link", type=float, default=1.0)
    parser.add_argument("--reward-separation", type=float, default=1.0)
    parser.add_argument("--reward-intervention", type=float, default=1.0)
    parser.add_argument("--reward-motion", type=float, default=1.0)
    parser.add_argument("--reward-failure", type=float, default=1.0)
    return parser


def main(algorithm: str) -> None:
    """Run the selected algorithm with one explicit seed shared by all RNG users."""
    args = _parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    weights = RewardWeights(args.reward_rate, args.reward_link, args.reward_separation, args.reward_intervention, args.reward_motion, args.reward_failure)
    environment_config = scenario_environment_config(
        MultiRelayEnvironment().config, num_relays=4, waypoint_radius_m=args.waypoint_radius,
        max_steps=args.max_steps, reward_weights=weights,
    )
    training_env = MultiRelayEnvironment(environment_config)
    evaluation_env = MultiRelayEnvironment(environment_config)
    observation, _ = training_env.reset(seed=args.seed)
    agent_class = ParameterSharingMATD3 if algorithm == "matd3" else ParameterSharingMADDPG
    agent = agent_class(observation["local"].shape[-1], observation["global"].shape[-1], environment_config.num_relays, device=args.device)
    replay_buffer = MultiAgentReplayBuffer(
        capacity=args.replay_capacity, num_relays=environment_config.num_relays,
        local_observation_dim=observation["local"].shape[-1], global_state_dim=observation["global"].shape[-1],
        action_dim=agent.action_dim, seed=args.seed,
    )
    training_config = DeterministicTrainingConfig(args.steps, args.replay_capacity, args.batch_size, args.random_action_steps, args.update_after_steps, args.updates_per_step, args.exploration_noise_std, args.seed)
    experiment_config = DeterministicExperimentConfig(args.output_dir, args.log_interval, args.evaluation_interval, args.evaluation_episodes, args.evaluation_seed, args.checkpoint_interval)
    result = run_deterministic_experiment(training_env, evaluation_env, agent, replay_buffer, training_config, experiment_config)
    run_config = json.loads((result.output_directory / "run_config.json").read_text(encoding="utf-8"))
    run_config["environment_config"] = {"tdma_mode": environment_config.tdma_mode, "antenna_mode": environment_config.antenna_mode, "waypoint_radius_m": args.waypoint_radius, "max_steps": environment_config.max_steps, "reward_weights": vars(weights)}
    (result.output_directory / "run_config.json").write_text(json.dumps(run_config, allow_nan=False, indent=2), encoding="utf-8")
    print(json.dumps({"algorithm": algorithm, "output_directory": str(result.output_directory), "best_mean_return": result.best_mean_return}, allow_nan=False))
