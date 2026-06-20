from __future__ import annotations

import argparse

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_sixdof_export import export_sixdof_puffer4_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the native Crazyflie 6-DoF env to a PufferLib 4 checkout")
    parser.add_argument("--pufferlib-root", required=True)
    parser.add_argument("--env-name", default="flightrl_sixdof")
    parser.add_argument("--total-agents", type=int, default=None)
    parser.add_argument("--num-buffers", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--policy-hidden-size", type=int, default=None)
    parser.add_argument("--policy-num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-profile", default=None)
    parser.add_argument("--task", default="position_yaw")
    parser.add_argument("--reward-mode", default="env")
    parser.add_argument("--reset-profile", default="broad")
    args = parser.parse_args()

    result = export_sixdof_puffer4_assets(
        args.pufferlib_root,
        settings=Puffer4ExportSettings(
            env_name=args.env_name,
            total_agents=args.total_agents,
            num_buffers=args.num_buffers,
            num_threads=args.num_threads,
            policy_hidden_size=args.policy_hidden_size,
            policy_num_layers=args.policy_num_layers,
            train_seed=args.seed,
            sim_profile=args.sim_profile,
            task=args.task,
            reward_mode=args.reward_mode,
            reset_profile=args.reset_profile,
        ),
    )
    print(f"env_dir={result.env_dir}")
    print(f"config_path={result.config_path}")


if __name__ == "__main__":
    main()
