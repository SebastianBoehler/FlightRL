from __future__ import annotations

import argparse
import json

from flightrl import load_config
from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_export import export_puffer4_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FlightRL as a PufferLib 4 environment")
    parser.add_argument("--config", required=True, help="Path to a FlightRL TOML config")
    parser.add_argument("--pufferlib-root", required=True, help="Path to a PufferLib 4.0 checkout")
    parser.add_argument("--env-name", default="flightrl", help="Target env name inside the PufferLib checkout")
    parser.add_argument("--total-agents", type=int, default=None, help="Override vec.total_agents")
    parser.add_argument("--num-buffers", type=int, default=None, help="Override vec.num_buffers")
    parser.add_argument("--num-threads", type=int, default=None, help="Override vec.num_threads")
    parser.add_argument("--policy-hidden-size", type=int, default=None, help="Override policy.hidden_size")
    parser.add_argument("--policy-num-layers", type=int, default=2, help="Set policy.num_layers")
    parser.add_argument("--train-seed", type=int, default=42, help="Seed written into the exported Puffer config")
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
    result = export_puffer4_assets(
        config=load_config(args.config),
        pufferlib_root=args.pufferlib_root,
        settings=settings,
    )
    print(
        json.dumps(
            {
                "env_name": result.env_name,
                "env_dir": str(result.env_dir),
                "config_path": str(result.config_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
