"""Generate reward and dynamic-scenario diagnostics as one JSON document."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.analysis.diagnostics import ScenarioDiagnosticConfig, diagnose_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radii", nargs="+", type=float, default=[30.0, 60.0, 90.0, 120.0])
    parser.add_argument("--max-steps", nargs="+", type=int, default=[100, 250])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--policies", nargs="+", default=["stationary", "equal_spacing"])
    parser.add_argument("--num-relays", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("output file already exists; refusing overwrite")
    base = MultiRelayEnvironment()
    base_config = replace(base.config, num_relays=args.num_relays)
    config = ScenarioDiagnosticConfig(tuple(args.radii), tuple(args.max_steps), args.episodes, args.seed, tuple(args.policies))
    result = diagnose_scenarios(base_config, config)
    payload = {
        "resolved_configuration": asdict(config),
        "episode_results": [asdict(item) for item in result.episode_results],
        "summaries": [asdict(item) for item in result.summaries],
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2)
    print(json.dumps({"output": str(output), "episodes": len(result.episode_results), "scenarios": len(result.summaries)}, allow_nan=False))


if __name__ == "__main__":
    main()
