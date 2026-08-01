from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from flightrl.hardware.ranger_projection import points_from_rows, prepare_rows, trajectory_from_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Crazyflie ranger telemetry as a sparse room point cloud")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-range-m", type=float, default=4.0)
    parser.add_argument("--min-drone-z-m", type=float, default=0.0)
    parser.add_argument("--raw-origin", action="store_true", help="keep estimator x/y instead of starting the plot at 0,0")
    args = parser.parse_args()

    input_path = Path(args.input)
    output = Path(args.output or input_path.with_suffix(".room.png"))
    with input_path.open() as handle:
        rows = prepare_rows(
            list(csv.DictReader(handle)),
            min_drone_z_m=args.min_drone_z_m,
            normalize_xy=not args.raw_origin,
        )
    points = points_from_rows(rows, max_range_m=args.max_range_m)
    trajectory = trajectory_from_rows(rows)
    if not points:
        raise SystemExit("no valid ranger points found")
    output.parent.mkdir(parents=True, exist_ok=True)
    render(points, trajectory, output)
    print(f"wrote {output} with {len(points)} ranger points and {len(trajectory)} trajectory samples")


def render(points, trajectory, output: Path) -> None:
    xs = [point.x_m for point in points]
    ys = [point.y_m for point in points]
    zs = [point.z_m for point in points]
    times = [point.time_s for point in points]
    drone_x = [pose.x_m for pose in trajectory]
    drone_y = [pose.y_m for pose in trajectory]
    drone_z = [pose.z_m for pose in trajectory]
    fig = plt.figure(figsize=(11, 5), constrained_layout=True)
    top = fig.add_subplot(1, 2, 1)
    top.scatter(xs, ys, s=4, c=times, cmap="viridis", alpha=0.5)
    top.plot(drone_x, drone_y, color="#20262e", linewidth=1.2, alpha=0.95, label="drone path")
    mark_endpoints(top, drone_x, drone_y)
    top.set_title("Top-down ranger map")
    top.set_xlabel("x m")
    top.set_ylabel("y m")
    top.legend(loc="best")
    top.axis("equal")
    side = fig.add_subplot(1, 2, 2, projection="3d")
    side.scatter(xs, ys, zs, s=3, c=times, cmap="viridis", alpha=0.42)
    side.plot(drone_x, drone_y, drone_z, color="#20262e", linewidth=1.3)
    mark_endpoints(side, drone_x, drone_y, drone_z)
    side.set_title("Sparse 3D point cloud + drone path")
    side.set_xlabel("x m")
    side.set_ylabel("y m")
    side.set_zlabel("z m")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def mark_endpoints(axis, xs, ys, zs=None) -> None:
    if not xs:
        return
    if zs is None:
        axis.scatter([xs[0]], [ys[0]], marker="o", s=35, color="#2ca25f", label="start")
        axis.scatter([xs[-1]], [ys[-1]], marker="x", s=45, color="#de2d26", label="end")
    else:
        axis.scatter([xs[0]], [ys[0]], [zs[0]], marker="o", s=35, color="#2ca25f")
        axis.scatter([xs[-1]], [ys[-1]], [zs[-1]], marker="x", s=45, color="#de2d26")


if __name__ == "__main__":
    main()
