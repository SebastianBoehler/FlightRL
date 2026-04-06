from __future__ import annotations

import argparse
import subprocess
import sys

from flightrl import load_config
from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_export import export_puffer4_assets


BUILD_MODE_FLAGS = {
    "default": [],
    "float": ["--float"],
    "cpu": ["--cpu"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FlightRL into PufferLib 4 and train it there")
    parser.add_argument("--config", required=True, help="Path to a FlightRL TOML config")
    parser.add_argument("--pufferlib-root", required=True, help="Path to a PufferLib 4.0 checkout")
    parser.add_argument("--env-name", default="flightrl", help="Target env name inside the PufferLib checkout")
    parser.add_argument("--build-mode", choices=tuple(BUILD_MODE_FLAGS), default="float")
    parser.add_argument("--no-build", action="store_true", help="Skip `build.sh` and only export before training")
    parser.add_argument("--total-agents", type=int, default=None)
    parser.add_argument("--num-buffers", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--policy-hidden-size", type=int, default=None)
    parser.add_argument("--policy-num-layers", type=int, default=2)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("puffer_args", nargs=argparse.REMAINDER, help="Arguments forwarded to `python -m pufferlib.pufferl train`")
    args = parser.parse_args()

    settings = Puffer4ExportSettings(
        env_name=args.env_name,
        total_agents=args.total_agents,
        num_buffers=args.num_buffers,
        num_threads=args.num_threads,
        policy_hidden_size=args.policy_hidden_size,
        policy_num_layers=args.policy_num_layers,
        train_seed=args.train_seed,
    )
    root = args.pufferlib_root
    export_puffer4_assets(load_config(args.config), root, settings=settings)

    if not args.no_build:
        subprocess.run(
            ["bash", "build.sh", args.env_name, *BUILD_MODE_FLAGS[args.build_mode]],
            cwd=root,
            check=True,
        )

    forwarded = list(args.puffer_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if args.build_mode == "cpu" and "--slowly" not in forwarded:
        forwarded = ["--slowly", *forwarded]
    subprocess.run(
        [sys.executable, "-m", "pufferlib.pufferl", "train", args.env_name, *forwarded],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
