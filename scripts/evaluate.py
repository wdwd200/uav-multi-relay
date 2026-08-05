"""Evaluate a saved MASAC checkpoint with deterministic actions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.training import MASACEvaluationConfig, evaluate_masac, load_masac_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    agent, metadata = load_masac_checkpoint(args.checkpoint, device=args.device)
    base = MultiRelayEnvironment()
    env = MultiRelayEnvironment(replace(base.config, num_relays=agent.num_relays))
    observation, _ = env.reset(seed=args.seed)
    if observation["local"].shape != (agent.num_relays, agent.local_observation_dim) or observation["global"].shape != (agent.global_state_dim,):
        raise ValueError("checkpoint dimensions do not match environment")
    summary = evaluate_masac(env, agent, MASACEvaluationConfig(args.episodes, args.seed))
    print(json.dumps({
        "episodes": summary.episodes,
        "mean_return": summary.mean_return,
        "return_std": summary.return_std,
        "mean_rate_e2e_bps": summary.mean_rate_e2e_bps,
        "minimum_rate_e2e_bps": summary.minimum_rate_e2e_bps,
        "mean_intervention_rate": summary.mean_intervention_rate,
        "terminated_episode_rate": summary.terminated_episode_rate,
        "checkpoint_environment_steps": metadata.environment_steps,
    }, allow_nan=False))


if __name__ == "__main__":
    main()
