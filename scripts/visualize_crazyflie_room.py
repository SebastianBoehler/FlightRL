from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from flightrl.hardware.ranger_map import points_from_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Crazyflie ranger telemetry as a sparse room point cloud")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-range-m", type=float, default=4.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output = Path(args.output or input_path.with_suffix(".room.png"))
    with input_path.open() as handle:
        rows = list(csv.DictReader(handle))
    points = points_from_rows(rows, max_range_m=args.max_range_m)
    if not points:
        raise SystemExit("no valid ranger points found")
    output.parent.mkdir(parents=True, exist_ok=True)
    render(points, output)
    print(f"wrote {output} with {len(points)} ranger points")


def render(points, output: Path) -> None:
    xs = [point.x_m for point in points]
    ys = [point.y_m for point in points]
    zs = [point.z_m for point in points]
    drone_x = [point.drone_x_m for point in points]
    drone_y = [point.drone_y_m for point in points]
    fig = plt.figure(figsize=(11, 5), constrained_layout=True)
    top = fig.add_subplot(1, 2, 1)
    top.scatter(xs, ys, s=4, c=zs, cmap="viridis", alpha=0.55)
    top.plot(drone_x, drone_y, color="#20262e", linewidth=1.0, alpha=0.9)
    top.set_title("Top-down ranger map")
    top.set_xlabel("x m")
    top.set_ylabel("y m")
    top.axis("equal")
    side = fig.add_subplot(1, 2, 2, projection="3d")
    side.scatter(xs, ys, zs, s=3, c=zs, cmap="viridis", alpha=0.5)
    side.set_title("Sparse 3D point cloud")
    side.set_xlabel("x m")
    side.set_ylabel("y m")
    side.set_zlabel("z m")
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
