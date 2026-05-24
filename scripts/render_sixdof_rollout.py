from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from flightrl.sixdof import BoxRoom
from flightrl.sixdof.render import render_rollout_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a 6-DoF rollout CSV as an interactive HTML scene")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    trajectory = load_trajectory(args.input)
    render_rollout_html(args.output, room=BoxRoom(), trajectory=trajectory)
    print(f"wrote {args.output}")
    return 0


def load_trajectory(path: Path) -> np.ndarray:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    points = []
    for row in rows:
        points.append([float(row.get("x", row.get("position.x", 0.0))), float(row.get("y", row.get("position.y", 0.0))), float(row.get("z", row.get("position.z", 0.0)))])
    return np.asarray(points, dtype=np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
