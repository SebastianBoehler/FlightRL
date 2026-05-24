from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from flightrl.hardware.manhattan_map import HORIZONTAL_SENSORS, box_corners_world, fit_manhattan_box, fit_to_dict, snap_points_to_box


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a Manhattan-world room box to sparse Crazyflie Multiranger points.")
    parser.add_argument("--points", required=True, help="CSV from export_crazyflie_room_scan.py")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--angle-samples", type=int, default=181)
    parser.add_argument("--quantile", type=float, default=0.03)
    parser.add_argument("--max-wall-residual-m", type=float, default=0.35)
    args = parser.parse_args()

    points_path = Path(args.points)
    if not points_path.exists():
        raise SystemExit(f"points CSV not found: {points_path}")
    rows = read_points(points_path)
    horizontal = np.asarray([[row["x_m"], row["y_m"], row["z_m"]] for row in rows if row["sensor"] in HORIZONTAL_SENSORS], dtype=np.float32)
    if len(horizontal) < 8:
        raise SystemExit(f"need at least 8 horizontal ranger points, found {len(horizontal)}")
    fit = fit_manhattan_box(horizontal, angle_samples=args.angle_samples, quantile=args.quantile, max_wall_residual_m=args.max_wall_residual_m)
    snapped = snap_points_to_box(horizontal, fit, max_wall_residual_m=args.max_wall_residual_m)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report = {**fit_to_dict(fit), "snapped_point_count": int(len(snapped)), "source": str(args.points)}
    write_json(prefix.with_suffix(".manhattan.json"), report)
    write_csv(prefix.with_suffix(".manhattan_points.csv"), snapped)
    write_ply(prefix.with_suffix(".manhattan.ply"), snapped)
    write_html(prefix.with_suffix(".manhattan.html"), horizontal, snapped, fit)
    print(f"report={prefix.with_suffix('.manhattan.json')}")
    print(f"html={prefix.with_suffix('.manhattan.html')}")


def read_points(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "sensor": str(row["sensor"]),
                "x_m": float(row["x_m"]),
                "y_m": float(row["y_m"]),
                "z_m": float(row["z_m"]),
            }
            for row in reader
        ]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, points: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_m", "y_m", "z_m"])
        writer.writerows(points.tolist())


def write_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def write_html(path: Path, raw: np.ndarray, snapped: np.ndarray, fit) -> None:
    corners = box_corners_world(fit)
    fig = go.Figure()
    fig.add_trace(scatter3d(raw, "raw horizontal hits", "rgba(120,120,120,0.35)", 2))
    fig.add_trace(scatter3d(snapped, "Manhattan-snapped wall hits", "#0072B2", 3))
    fig.add_trace(
        go.Scatter3d(
            x=corners[:, 0],
            y=corners[:, 1],
            z=np.full(len(corners), fit.z_min),
            mode="lines",
            name="estimated floor box",
            line={"color": "#D55E00", "width": 6},
        )
    )
    fig.update_layout(scene={"aspectmode": "data"}, margin={"l": 0, "r": 0, "t": 30, "b": 0}, title="Crazyflie Manhattan Room Fit")
    fig.write_html(path, include_plotlyjs="cdn")


def scatter3d(points: np.ndarray, name: str, color: str, size: int):
    return go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", name=name, marker={"size": size, "color": color})


if __name__ == "__main__":
    main()
