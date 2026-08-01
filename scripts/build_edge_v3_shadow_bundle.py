from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.artifact_paths import require_distinct_artifact_paths
from flightrl.puffer4_edge_shadow_bundle import (
    build_edge_shadow_bundle,
    write_edge_shadow_bundle,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed offline passive-shadow bundle"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/edge_v3/edge_door_shadow_bundle.json"),
    )
    args = parser.parse_args(argv)
    paths = require_distinct_artifact_paths(
        checkpoint=args.checkpoint,
        evaluation=args.evaluation_report,
        replay=args.replay,
        output=args.output,
    )
    bundle = build_edge_shadow_bundle(
        checkpoint=paths["checkpoint"],
        evaluation_report=paths["evaluation"],
        replay=paths["replay"],
    )
    write_edge_shadow_bundle(bundle, paths["output"])
    print(f"bundle={paths['output']}")


if __name__ == "__main__":
    main()
