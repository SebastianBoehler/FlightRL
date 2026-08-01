from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_export import export_fixed_door_assets
from flightrl.puffer4_door_runner import build_environment, load_puffer
from flightrl.puffer4_door_training import (
    evaluate_door_teacher,
    fixed_door_teacher_gate,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUFFER = ROOT.parent / "PufferLib-4-flightrl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the observation-matched fixed-door teacher"
    )
    parser.add_argument("--puffer-root", type=Path, default=DEFAULT_PUFFER)
    parser.add_argument("--env-name", default="flightrl_fixed_door_d1")
    parser.add_argument("--agents", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=10_011)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    settings = Puffer4ExportSettings(
        env_name=args.env_name,
        total_agents=args.agents,
        num_buffers=1,
        num_threads=8,
        policy_hidden_size=96,
        policy_num_layers=1,
        train_seed=11,
    )
    export_fixed_door_assets(args.puffer_root, settings)
    if not args.skip_build:
        build_environment(args.puffer_root, args.env_name)
    puffer_args, torch_pufferl = load_puffer(args.puffer_root, args.env_name)
    puffer_args["env"]["obstacle_probability"] = 0.0
    puffer_args["env"]["layout_diversity"] = 1.0
    puffer_args["vec"]["total_agents"] = args.agents
    metrics = evaluate_door_teacher(
        puffer_args,
        torch_pufferl,
        steps=args.steps,
        seed=args.seed,
        agents=args.agents,
    )
    print(
        json.dumps(
            {"metrics": metrics, "gate": fixed_door_teacher_gate(metrics)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
