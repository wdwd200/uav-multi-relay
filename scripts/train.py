"""Minimal command-line MASAC training entry point."""

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
from uav_multi_relay.training import MASACCheckpointMetadata, MASACTrainingConfig, save_masac_checkpoint, train_masac


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-relays", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--random-action-steps", type=int, default=1_000)
    parser.add_argument("--update-after-steps", type=int, default=1_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-out")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    base = MultiRelayEnvironment()
    env = MultiRelayEnvironment(replace(base.config, num_relays=args.num_relays))
    observation, _ = env.reset(seed=args.seed)
    local_dim = int(observation["local"].shape[-1])
    global_dim = int(observation["global"].shape[-1])
    replay = MultiAgentReplayBuffer(
        capacity=max(args.batch_size, 100_000),
        num_relays=args.num_relays,
        local_observation_dim=local_dim,
        global_state_dim=global_dim,
        action_dim=3,
        seed=args.seed,
    )
    agent = ParameterSharingMASAC(
        local_observation_dim=local_dim,
        global_state_dim=global_dim,
        num_relays=args.num_relays,
        action_dim=3,
        device=args.device,
    )
    config = MASACTrainingConfig(
        total_environment_steps=args.steps,
        replay_capacity=replay.capacity,
        batch_size=args.batch_size,
        random_action_steps=args.random_action_steps,
        update_after_steps=args.update_after_steps,
        updates_per_step=args.updates_per_step,
        seed=args.seed,
    )
    summary = train_masac(env, agent, replay, config)
    result = {
        "total_environment_steps": summary.total_environment_steps,
        "total_updates": summary.total_updates,
        "completed_episodes": summary.completed_episodes,
        "mean_rate_e2e_bps": summary.mean_rate_e2e_bps,
        "intervention_rate": summary.intervention_rate,
    }
    if args.checkpoint_out:
        checkpoint_path = save_masac_checkpoint(
            args.checkpoint_out,
            agent,
            MASACCheckpointMetadata(summary.total_environment_steps, summary.total_updates, summary.completed_episodes),
        )
        result["checkpoint_path"] = str(checkpoint_path)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
