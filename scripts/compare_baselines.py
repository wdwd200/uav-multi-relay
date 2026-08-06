"""Compare a MASAC checkpoint with deterministic and random relay baselines."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uav_multi_relay import MultiRelayEnvironment, RewardWeights, scenario_environment_config
from uav_multi_relay.analysis import PolicyComparisonConfig, compare_policies
from uav_multi_relay.policies import MPCConfig
from uav_multi_relay.training import load_masac_checkpoint


def _prepare_output(directory: Path) -> None:
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("output directory must be missing or empty")
    directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--waypoint-radius", type=float, default=30.0)
    parser.add_argument("--policies", nargs="+", default=["masac", "random", "stationary", "equal_spacing", "weighted_spacing", "greedy", "mpc"])
    parser.add_argument("--greedy-sweeps", type=int, default=1)
    parser.add_argument("--mpc-horizon", type=int, default=2)
    parser.add_argument("--mpc-population-size", type=int, default=8)
    parser.add_argument("--mpc-iterations", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reward-rate", type=float, default=1.0)
    parser.add_argument("--reward-link", type=float, default=1.0)
    parser.add_argument("--reward-separation", type=float, default=1.0)
    parser.add_argument("--reward-intervention", type=float, default=1.0)
    parser.add_argument("--reward-motion", type=float, default=1.0)
    parser.add_argument("--reward-failure", type=float, default=1.0)
    args = parser.parse_args()
    agent, _ = load_masac_checkpoint(args.checkpoint, device=args.device)
    base = MultiRelayEnvironment()
    reward_weights = RewardWeights(
        args.reward_rate, args.reward_link, args.reward_separation,
        args.reward_intervention, args.reward_motion, args.reward_failure,
    )
    environment_config = scenario_environment_config(
        base.config,
        num_relays=agent.num_relays,
        waypoint_radius_m=args.waypoint_radius,
        max_steps=args.max_steps,
        reward_weights=reward_weights,
    )
    env = MultiRelayEnvironment(environment_config)
    observation, _ = env.reset(seed=args.seed)
    if observation["local"].shape != (agent.num_relays, agent.local_observation_dim) or observation["global"].shape != (agent.global_state_dim,):
        raise ValueError("checkpoint dimensions do not match environment")
    mpc_config = MPCConfig(args.mpc_horizon, args.mpc_population_size, args.mpc_iterations, 0.5)
    config = PolicyComparisonConfig(args.episodes, args.seed, tuple(args.policies), args.greedy_sweeps, mpc_config)
    output = Path(args.output_dir)
    _prepare_output(output)
    result = compare_policies(env, agent, config)
    with (output / "comparison_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"comparison_config": asdict(config), "agent": {"num_relays": agent.num_relays, "local_observation_dim": agent.local_observation_dim, "global_state_dim": agent.global_state_dim, "action_dim": agent.action_dim}, "environment_config": {"waypoint_radius_m": args.waypoint_radius, "max_steps": environment_config.max_steps, "reward_weights": vars(reward_weights)}}, handle, allow_nan=False, indent=2)
    with (output / "comparison_episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in result.episode_results:
            handle.write(json.dumps(asdict(episode), allow_nan=False) + "\n")
    with (output / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"policy_summaries": [asdict(summary) for summary in result.policy_summaries]}, handle, allow_nan=False, indent=2)
    print(json.dumps({"output_directory": str(output), "policies": [summary.policy for summary in result.policy_summaries], "mean_returns": {summary.policy: summary.mean_return for summary in result.policy_summaries}}, allow_nan=False))


if __name__ == "__main__":
    main()
