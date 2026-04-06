from __future__ import annotations

import argparse

from .config import load_config
from .puffer4_config import Puffer4ExportSettings
from .puffer4_runtime import BUILD_MODE_FLAGS, run_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FlightRL through an upstream PufferLib 4 checkout")
    parser.add_argument("--config", required=True, help="Path to a FlightRL TOML config")
    parser.add_argument("--pufferlib-root", default=None, help="Path to a PufferLib 4 checkout")
    parser.add_argument("--env-name", default="flightrl", help="Target env name inside the PufferLib checkout")
    parser.add_argument("--build-mode", choices=tuple(BUILD_MODE_FLAGS), default="float")
    parser.add_argument("--no-build", action="store_true", help="Skip `build.sh` and only export before training")
    parser.add_argument("--total-agents", type=int, default=None)
    parser.add_argument("--num-buffers", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--policy-hidden-size", type=int, default=None)
    parser.add_argument("--policy-num-layers", type=int, default=2)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument(
        "puffer_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to `python -m pufferlib.pufferl train`",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Puffer4ExportSettings(
        env_name=args.env_name,
        total_agents=args.total_agents,
        num_buffers=args.num_buffers,
        num_threads=args.num_threads,
        policy_hidden_size=args.policy_hidden_size,
        policy_num_layers=args.policy_num_layers,
        train_seed=args.train_seed,
    )
    run_train(
        load_config(args.config),
        pufferlib_root=args.pufferlib_root,
        settings=settings,
        build_mode=args.build_mode,
        no_build=args.no_build,
        puffer_args=args.puffer_args,
    )
