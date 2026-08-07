"""Compare selected MAPPO/MASAC checkpoints with deterministic relay baselines."""

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
from uav_multi_relay.training import load_maddpg_checkpoint, load_mappo_checkpoint, load_masac_checkpoint, load_matd3_checkpoint


def _prepare_output(directory: Path) -> None:
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("output directory must be missing or empty")
    directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Legacy alias for --masac-checkpoint.")
    parser.add_argument("--masac-checkpoint")
    parser.add_argument("--mappo-checkpoint")
    parser.add_argument("--matd3-checkpoint")
    parser.add_argument("--maddpg-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--waypoint-radius", type=float, default=30.0)
    parser.add_argument("--tdma-mode", choices=("optimal", "equal"), default="optimal")
    parser.add_argument("--antenna-mode", choices=("dipole", "isotropic"), default="dipole")
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
    if args.checkpoint is not None and args.masac_checkpoint is not None:
        parser.error("use either --checkpoint or --masac-checkpoint, not both")
    masac_path = args.masac_checkpoint or args.checkpoint
    requested = set(args.policies)
    if "masac" in requested and masac_path is None:
        parser.error("MASAC policy requires --masac-checkpoint (or legacy --checkpoint)")
    if "mappo" in requested and args.mappo_checkpoint is None:
        parser.error("MAPPO policy requires --mappo-checkpoint")
    if "matd3" in requested and args.matd3_checkpoint is None: parser.error("MATD3 policy requires --matd3-checkpoint")
    if "maddpg" in requested and args.maddpg_checkpoint is None: parser.error("MADDPG policy requires --maddpg-checkpoint")
    masac_agent = load_masac_checkpoint(masac_path, device=args.device)[0] if "masac" in requested else None
    mappo_agent = load_mappo_checkpoint(args.mappo_checkpoint, device=args.device)[0] if "mappo" in requested else None
    matd3_agent = load_matd3_checkpoint(args.matd3_checkpoint, device=args.device)[0] if "matd3" in requested else None
    maddpg_agent = load_maddpg_checkpoint(args.maddpg_checkpoint, device=args.device)[0] if "maddpg" in requested else None
    reference = mappo_agent or masac_agent or matd3_agent or maddpg_agent
    num_relays = reference.num_relays if reference is not None else 4
    base = MultiRelayEnvironment()
    reward_weights = RewardWeights(
        args.reward_rate, args.reward_link, args.reward_separation,
        args.reward_intervention, args.reward_motion, args.reward_failure,
    )
    environment_config = scenario_environment_config(
        base.config,
        num_relays=num_relays,
        waypoint_radius_m=args.waypoint_radius,
        max_steps=args.max_steps,
        reward_weights=reward_weights,
        tdma_mode=args.tdma_mode,
        antenna_mode=args.antenna_mode,
    )
    env = MultiRelayEnvironment(environment_config)
    observation, _ = env.reset(seed=args.seed)
    for name, agent in (("MASAC", masac_agent), ("MAPPO", mappo_agent), ("MATD3", matd3_agent), ("MADDPG", maddpg_agent)):
        if agent is not None and (observation["local"].shape != (agent.num_relays, agent.local_observation_dim) or observation["global"].shape != (agent.global_state_dim,)):
            raise ValueError(f"{name} checkpoint dimensions do not match environment")
    agents = [item for item in (masac_agent,mappo_agent,matd3_agent,maddpg_agent) if item is not None]
    if any((item.num_relays,item.local_observation_dim,item.global_state_dim)!=(agents[0].num_relays,agents[0].local_observation_dim,agents[0].global_state_dim) for item in agents): raise ValueError("learning checkpoint dimensions are incompatible")
    mpc_config = MPCConfig(args.mpc_horizon, args.mpc_population_size, args.mpc_iterations, 0.5)
    config = PolicyComparisonConfig(args.episodes, args.seed, tuple(args.policies), args.greedy_sweeps, mpc_config)
    output = Path(args.output_dir)
    _prepare_output(output)
    result = compare_policies(env, masac_agent, config, mappo_agent=mappo_agent, matd3_agent=matd3_agent, maddpg_agent=maddpg_agent)
    with (output / "comparison_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"comparison_config": asdict(config), "agents": {name: None if agent is None else {"num_relays": agent.num_relays, "local_observation_dim": agent.local_observation_dim, "global_state_dim": agent.global_state_dim, "action_dim": agent.action_dim} for name, agent in (("mappo", mappo_agent), ("masac", masac_agent), ("matd3",matd3_agent),("maddpg",maddpg_agent))}, "environment_config": {"waypoint_radius_m": args.waypoint_radius, "max_steps": environment_config.max_steps, "tdma_mode": environment_config.tdma_mode, "antenna_mode": environment_config.antenna_mode, "reward_weights": vars(reward_weights)}}, handle, allow_nan=False, indent=2)
    with (output / "comparison_episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in result.episode_results:
            handle.write(json.dumps(asdict(episode), allow_nan=False) + "\n")
    with (output / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"policy_summaries": [asdict(summary) for summary in result.policy_summaries]}, handle, allow_nan=False, indent=2)
    print(json.dumps({"output_directory": str(output), "policies": [summary.policy for summary in result.policy_summaries], "mean_returns": {summary.policy: summary.mean_return for summary in result.policy_summaries}}, allow_nan=False))


if __name__ == "__main__":
    main()
