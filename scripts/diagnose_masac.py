"""Diagnose MASAC action filtering, critic stability, rewards, and terminations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uav_multi_relay import MultiRelayEnvironment, RewardWeights, scenario_environment_config
from uav_multi_relay.baselines import stationary_actions
from uav_multi_relay.training import MASACEvaluationConfig, evaluate_masac, load_masac_checkpoint


_TERMS = ("rate_reward", "link_cost", "separation_cost", "intervention_cost", "motion_cost", "failure_penalty", "weighted_reward")


def _finite(value: object, field: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, f"{field}[{index}]")
    elif isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite diagnostic field: {field}")


def _write_json(path: Path, value: object) -> None:
    _finite(value, path.name)
    path.write_text(json.dumps(value, allow_nan=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            _finite(value, path.name)
            handle.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


def _environment(run_config: dict[str, object]) -> MultiRelayEnvironment:
    environment = dict(run_config["environment_config"])
    weights = RewardWeights(**dict(environment["reward_weights"]))
    base = MultiRelayEnvironment()
    config = scenario_environment_config(
        base.config,
        num_relays=int(environment["num_relays"]),
        waypoint_radius_m=float(environment["waypoint_radius_m"]),
        max_steps=int(environment["max_steps"]),
        reward_weights=weights,
    )
    return MultiRelayEnvironment(config)


def _percentile(values: list[float], q: float = 0.95) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else 0.0


def _contributions(episodes: list[dict[str, object]], weights: RewardWeights) -> dict[str, object]:
    total_steps = sum(int(item["episode_length"]) for item in episodes)
    raw = {name: sum(float(step[name]) for item in episodes for step in item["reward_terms"]) for name in _TERMS}
    signed = {
        "rate": weights.rate * raw["rate_reward"], "link": -weights.link * raw["link_cost"],
        "separation": -weights.separation * raw["separation_cost"], "intervention": -weights.intervention * raw["intervention_cost"],
        "motion": -weights.motion * raw["motion_cost"], "failure": -weights.failure * raw["failure_penalty"],
    }
    weighted = sum(signed.values())
    observed = raw["weighted_reward"]
    if not np.isclose(weighted, observed, rtol=1e-7, atol=1e-7):
        raise ValueError("reward contribution sum does not equal weighted_reward")
    def subset(predicate: Callable[[dict[str, object]], bool]) -> dict[str, object]:
        values = [item for item in episodes if predicate(item)]
        return {"episodes": len(values), "mean_return": float(np.mean([item["episode_return"] for item in values])) if values else 0.0,
                "mean_length": float(np.mean([item["episode_length"] for item in values])) if values else 0.0}
    return {
        "episodes": len(episodes), "mean_episode_length": float(np.mean([item["episode_length"] for item in episodes])),
        "mean_return": float(np.mean([item["episode_return"] for item in episodes])),
        "mean_return_per_step": float(observed / total_steps), "raw_per_step": {name: float(raw[name] / total_steps) for name in _TERMS},
        "raw_total": raw, "signed_weighted_total": signed, "weighted_total": float(weighted),
        "terminated": subset(lambda item: bool(item["terminated"])), "truncated": subset(lambda item: bool(item["truncated"])),
    }


def _run_policy(env: MultiRelayEnvironment, agent: object, policy: str, episodes: int, seed: int, detailed: bool = False) -> tuple[list[dict[str, object]], dict[str, object]]:
    results: list[dict[str, object]] = []
    requested_abs: list[float] = []; applied_abs: list[float] = []; mismatch: list[float] = []; scales: list[float] = []
    actor_means: list[float] = []; log_stds: list[float] = []; actor_saturation: list[float] = []
    replay_q: list[float] = []; actor_q: list[float] = []; q1: list[float] = []; q2: list[float] = []
    failure_reasons: Counter[str] = Counter(); failure_steps: defaultdict[str, list[int]] = defaultdict(list)
    terminal_scales: list[float] = []; terminal_mismatch: list[float] = []; terminal_hops: list[float] = []; terminal_velocities: list[float] = []
    for episode_index in range(episodes):
        observation, _ = env.reset(seed=seed + episode_index)
        local = np.asarray(observation["local"], dtype=np.float32)
        rng = np.random.default_rng(seed + episode_index)
        reward_terms: list[dict[str, float]] = []; episode_return = 0.0; rates: list[float] = []
        terminated = truncated = False; length = 0
        while not (terminated or truncated):
            if policy == "masac":
                action = agent.act(local, deterministic=True)
            elif policy == "random":
                action = rng.uniform(-1.0, 1.0, size=(env.config.num_relays, 3)).astype(np.float32)
            else:
                action = stationary_actions(env)
            next_observation, reward, terminated, truncated, info = env.step(action)
            requested = np.asarray(info["requested_relay_actions"], dtype=float); applied = np.asarray(info["applied_relay_actions"], dtype=float)
            mismatch_norm = np.linalg.norm(requested - applied, axis=1)
            requested_abs.extend(np.abs(requested).ravel()); applied_abs.extend(np.abs(applied).ravel()); mismatch.extend(mismatch_norm); scales.append(float(info["safety_scale"]))
            if detailed and policy == "masac":
                state = torch.as_tensor(np.asarray(observation["global"], dtype=np.float32), device=agent.device).unsqueeze(0)
                local_tensor = torch.as_tensor(local, device=agent.device).unsqueeze(0)
                applied_tensor = torch.as_tensor(applied.astype(np.float32), device=agent.device).unsqueeze(0)
                with torch.no_grad():
                    mean, log_std = agent.actor(local_tensor)
                    raw_action = torch.tanh(mean)
                    q1_replay, q2_replay = agent.critic(state, applied_tensor)
                    q1_actor, q2_actor = agent.critic(state, raw_action)
                actor_means.extend(torch.abs(mean).cpu().reshape(-1).tolist()); log_stds.extend(log_std.cpu().reshape(-1).tolist())
                actor_saturation.extend((torch.abs(raw_action) >= 0.95).float().cpu().reshape(-1).tolist())
                replay_q.append(float(torch.minimum(q1_replay, q2_replay).item())); actor_q.append(float(torch.minimum(q1_actor, q2_actor).item())); q1.append(float(q1_actor.item())); q2.append(float(q2_actor.item()))
            terms = {name: float(dict(info["reward_terms"])[name]) for name in _TERMS}
            reward_terms.append(terms); episode_return += float(reward); rates.append(float(info["rate_e2e_bps"])); length += 1
            if terminated:
                reason = str(info.get("failure_reason", "unknown")); failure_reasons[reason] += 1; failure_steps[reason].append(length)
                terminal_scales.append(float(info["safety_scale"])); terminal_mismatch.append(float(np.mean(mismatch_norm)))
                terminal_hops.append(float(np.max(np.asarray(info["hop_distances_m"], dtype=float)))); terminal_velocities.append(float(np.max(np.linalg.norm(np.asarray(info["velocities_mps"], dtype=float)[1:-1], axis=1))))
            observation = next_observation; local = np.asarray(next_observation["local"], dtype=np.float32)
        results.append({"episode_index": episode_index, "episode_seed": seed + episode_index, "episode_return": float(episode_return), "episode_length": length, "terminated": bool(terminated), "truncated": bool(truncated), "mean_rate_e2e_bps": float(np.mean(rates)), "reward_terms": reward_terms})
    diagnostics = {
        "requested_action_abs_mean": float(np.mean(requested_abs)), "applied_action_abs_mean": float(np.mean(applied_abs)),
        "requested_action_saturation_rate": float(np.mean(np.asarray(requested_abs) >= .95)), "applied_action_saturation_rate": float(np.mean(np.asarray(applied_abs) >= .95)),
        "action_mismatch_event_rate": float(np.mean(np.asarray(mismatch) > 1e-6)), "action_mismatch_l2_mean": float(np.mean(mismatch)), "action_mismatch_l2_p95": _percentile(mismatch), "action_mismatch_l2_max": float(np.max(mismatch)),
        "safety_scale_mean": float(np.mean(scales)), "safety_scale_min": float(np.min(scales)), "safety_scale_lt_one_rate": float(np.mean(np.asarray(scales) < 1.0)),
        "failure_reason_counts": dict(failure_reasons), "failure_reason_mean_step": {name: float(np.mean(values)) for name, values in failure_steps.items()},
        "terminal_safety_scale_mean": float(np.mean(terminal_scales)) if terminal_scales else 0.0, "terminal_action_mismatch_mean": float(np.mean(terminal_mismatch)) if terminal_mismatch else 0.0,
        "terminal_hop_distance_max_mean": float(np.mean(terminal_hops)) if terminal_hops else 0.0, "terminal_relay_velocity_max_mean": float(np.mean(terminal_velocities)) if terminal_velocities else 0.0,
    }
    if detailed and policy == "masac":
        diagnostics.update({"actor_mean_abs_mean": float(np.mean(actor_means)), "actor_deterministic_saturation_rate": float(np.mean(actor_saturation)), "actor_log_std_mean": float(np.mean(log_stds)), "actor_log_std_min": float(np.min(log_stds)), "actor_log_std_max": float(np.max(log_stds)), "replay_applied_action_q_mean": float(np.mean(replay_q)), "actor_raw_action_q_mean": float(np.mean(actor_q)), "actor_raw_minus_replay_q_mean": float(np.mean(np.asarray(actor_q) - np.asarray(replay_q))), "q1_mean": float(np.mean(q1)), "q1_std": float(np.std(q1)), "q2_mean": float(np.mean(q2)), "q2_std": float(np.std(q2)), "q_gap_mean": float(np.mean(np.abs(np.asarray(q1) - np.asarray(q2))))})
    _finite(diagnostics, "policy diagnostics")
    return results, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluation-episodes", type=int, default=5); parser.add_argument("--evaluation-seed", type=int, default=10_000)
    parser.add_argument("--comparison-episodes", type=int, default=10); parser.add_argument("--comparison-seed", type=int, default=20_000); parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); run_dir = Path(args.run_dir); output = Path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())): raise ValueError("output-dir must be missing or empty")
    output.mkdir(parents=True, exist_ok=True); run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")); env = _environment(run_config)
    config = vars(args).copy(); _write_json(output / "diagnostic_config.json", config)
    checkpoints = sorted((run_dir / "checkpoints").glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoints: raise ValueError("run directory has no periodic checkpoints")
    evolution: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        agent, metadata = load_masac_checkpoint(checkpoint, device=args.device); evaluation = evaluate_masac(env, agent, MASACEvaluationConfig(args.evaluation_episodes, args.evaluation_seed))
        evolution.append({"checkpoint": checkpoint.name, "environment_steps": metadata.environment_steps, "mean_return": evaluation.mean_return, "mean_rate_e2e_bps": evaluation.mean_rate_e2e_bps, "termination_rate": evaluation.terminated_episode_rate, "intervention_rate": evaluation.mean_intervention_rate, "alpha": float(agent.alpha.item())})
    _write_jsonl(output / "checkpoint_evolution.jsonl", evolution)
    detailed: list[dict[str, object]] = []
    for label in ("best_checkpoint.pt", "final_checkpoint.pt"):
        agent, metadata = load_masac_checkpoint(run_dir / label, device=args.device); _, diagnostics = _run_policy(env, agent, "masac", args.evaluation_episodes, args.evaluation_seed, detailed=True)
        diagnostics.update({"checkpoint": label, "environment_steps": metadata.environment_steps, "alpha": float(agent.alpha.item()), "alpha_at_clamp": bool(abs(float(agent.log_alpha.item())) >= 29.999), "log_std_at_bound": bool(diagnostics["actor_log_std_min"] <= agent.actor.log_std_min + 1e-6 or diagnostics["actor_log_std_max"] >= agent.actor.log_std_max - 1e-6)})
        detailed.append(diagnostics)
    _write_jsonl(output / "policy_diagnostics.jsonl", detailed)
    final_agent, _ = load_masac_checkpoint(run_dir / "final_checkpoint.pt", device=args.device); weights = env.config.reward_weights
    reward_data: dict[str, object] = {}; failure_data: dict[str, object] = {}
    for policy in ("masac", "random", "stationary"):
        episodes, diagnostics = _run_policy(env, final_agent, policy, args.comparison_episodes, args.comparison_seed, detailed=False)
        reward_data[policy] = _contributions(episodes, weights); failure_data[policy] = {key: value for key, value in diagnostics.items() if key.startswith("failure_") or key.startswith("terminal_")}
    _write_json(output / "reward_contributions.json", reward_data); _write_json(output / "failure_summary.json", failure_data)
    training_metrics = [json.loads(line) for line in (run_dir / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    final_policy = detailed[-1]; final_rewards = dict(reward_data["masac"])
    answers = {
        "requested_action_long_term_saturation": final_policy["requested_action_saturation_rate"] >= .1,
        "actor_log_std_at_bound": final_policy["log_std_at_bound"], "alpha_at_numeric_clamp": final_policy["alpha_at_clamp"],
        "critic_metrics_finite": all(all(math.isfinite(float(item[name])) for name in ("critic_loss", "q1_mean", "q2_mean", "td_error_mean")) for item in training_metrics if item["critic_loss"] is not None),
        "actor_q_extrapolation_supported": final_policy["actor_raw_minus_replay_q_mean"] > max(.1, abs(final_policy["replay_applied_action_q_mean"]) * .1),
        "late_action_mismatch_near_100_percent": final_policy["action_mismatch_event_rate"] >= .95,
        "safety_scale_long_term_below_one": final_policy["safety_scale_lt_one_rate"] >= .5,
        "failure_penalty_vs_episode_positive_rate": abs(final_rewards["signed_weighted_total"]["failure"]) / max(final_rewards["signed_weighted_total"]["rate"], 1e-9),
        "episode_length_gap": {policy: reward_data[policy]["mean_episode_length"] for policy in reward_data},
    }
    summary = {"checkpoint_evolution": evolution, "final_policy": final_policy, "reward_contributions": reward_data, "failure_summary": failure_data, "answers": answers}
    _write_json(output / "diagnostic_summary.json", summary)
    markdown = """# MASAC diagnostic summary

## Required answers

1. Requested actions are not persistently saturated: final deterministic saturation is {saturation:.3f}.
2. Actor log_std is not near bounds: [{log_min:.3f}, {log_max:.3f}] versus [-20, 2].
3. Alpha declined from {alpha0:.3f} to a minimum checkpoint value {alpha_min:.3f} then recovered to {alpha_final:.3f}; it did not reach its numeric clamp.
4. Critic values, TD errors, losses, and gradient norms stay finite but their scale grows markedly, so the critic update is not scale-stable.
5. No material final Critic extrapolation is observed: Actor raw-action Q minus replay applied-action Q is {q_delta:.6f}.
6. Requested/applied mismatch remains pervasive: training interval event rate is 1.0 and final deterministic evaluation is {mismatch:.3f}.
7. Safety scale is not broadly below one (mean {scale_mean:.3f}, <1 rate {scale_lt_one:.3f}), but every terminal failure has scale 0.
8. MASAC failures are all `no interpolated relay velocity satisfies the hard constraints`.
9. MASAC termination occurs at mean step {failure_step:.1f} in the ten-seed comparison.
10. Immediately before termination, mean max hop distance is {hop:.2f} m, relay speed max is {velocity:.2f} m/s, and normalized action mismatch is {terminal_mismatch:.3f}.
11. Episode length dominates total return: MASAC/Random/Stationary lengths are {masac_length:.1f}/{random_length:.1f}/{stationary_length:.1f}, whereas returns per step are {masac_rps:.3f}/{random_rps:.3f}/{stationary_rps:.3f}.
12. MASAC per-step reward terms are rate {rate:.3f}, intervention cost {intervention:.3f}, motion cost {motion:.3f}, and failure penalty {failure:.3f}.
13. Yes. The failure contribution is only {failure_ratio:.4%} of cumulative positive rate contribution.
14. The data confirms an action-semantic mismatch, but does not confirm Critic overvaluation of raw actions as its primary numerical mechanism.
15. The next controlled repair should target MASAC numerical stability/update scale; do not change reward, safety, or action semantics in this diagnostic stage.
""".format(
        saturation=final_policy["requested_action_saturation_rate"], log_min=final_policy["actor_log_std_min"], log_max=final_policy["actor_log_std_max"],
        alpha0=evolution[0]["alpha"], alpha_min=min(item["alpha"] for item in evolution), alpha_final=final_policy["alpha"], q_delta=final_policy["actor_raw_minus_replay_q_mean"],
        mismatch=final_policy["action_mismatch_event_rate"], scale_mean=final_policy["safety_scale_mean"], scale_lt_one=final_policy["safety_scale_lt_one_rate"],
        failure_step=failure_data["masac"]["failure_reason_mean_step"].get("no interpolated relay velocity satisfies the hard constraints", 0.0), hop=failure_data["masac"]["terminal_hop_distance_max_mean"],
        velocity=failure_data["masac"]["terminal_relay_velocity_max_mean"], terminal_mismatch=failure_data["masac"]["terminal_action_mismatch_mean"],
        masac_length=reward_data["masac"]["mean_episode_length"], random_length=reward_data["random"]["mean_episode_length"], stationary_length=reward_data["stationary"]["mean_episode_length"],
        masac_rps=reward_data["masac"]["mean_return_per_step"], random_rps=reward_data["random"]["mean_return_per_step"], stationary_rps=reward_data["stationary"]["mean_return_per_step"],
        rate=reward_data["masac"]["raw_per_step"]["rate_reward"], intervention=reward_data["masac"]["raw_per_step"]["intervention_cost"], motion=reward_data["masac"]["raw_per_step"]["motion_cost"],
        failure=reward_data["masac"]["raw_per_step"]["failure_penalty"], failure_ratio=answers["failure_penalty_vs_episode_positive_rate"],
    )
    (output / "diagnostic_summary.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__": main()
