from __future__ import annotations

import argparse
import json

from flightrl import load_config
from flightrl.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FlightRL with PuffeRL")
    parser.add_argument("--config", required=True, help="Path to a TOML config file")
    args = parser.parse_args()

    logs = run_training(load_config(args.config))
    print(json.dumps(logs[-1] if logs else {}, indent=2))


if __name__ == "__main__":
    main()
