from __future__ import annotations

import argparse

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_sixdof_runtime import run_sixdof_train
from flightrl.puffer4_runtime import BUILD_MODE_FLAGS


def main() -> None:
    parser = argparse.ArgumentParser(description="Export/build/train the 6-DoF Crazyflie env in a PufferLib 4 checkout")
    parser.add_argument("--pufferlib-root", required=True, help="Path to a PufferLib 4 checkout")
    parser.add_argument("--env-name", default="flightrl_sixdof")
    parser.add_argument("--total-agents", type=int, default=4096)
    parser.add_argument("--num-buffers", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--policy-hidden-size", type=int, default=None)
    parser.add_argument("--policy-num-layers", type=int, default=2)
    parser.add_argument("--build-mode", choices=tuple(BUILD_MODE_FLAGS), default="cpu")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("puffer_args", nargs=argparse.REMAINDER, help="Arguments forwarded after optional --")
    args = parser.parse_args()

    run_sixdof_train(
        pufferlib_root=args.pufferlib_root,
        settings=Puffer4ExportSettings(
            env_name=args.env_name,
            total_agents=args.total_agents,
            num_buffers=args.num_buffers,
            num_threads=args.num_threads,
            policy_hidden_size=args.policy_hidden_size,
            policy_num_layers=args.policy_num_layers,
            train_seed=args.train_seed,
        ),
        build_mode=args.build_mode,
        no_build=args.no_build,
        puffer_args=args.puffer_args,
    )


if __name__ == "__main__":
    main()
