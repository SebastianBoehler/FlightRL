from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.puffer4_edge_replay import write_edge_passive_replay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run non-actuating offline edge-v3 recurrent shadow replay"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/edge_v3/edge_door_shadow.jsonl"),
    )
    args = parser.parse_args()

    summary = write_edge_passive_replay(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        output=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
