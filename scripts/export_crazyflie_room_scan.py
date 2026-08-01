from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import plotly.graph_objects as go

from flightrl.hardware.ranger_integrity import ranger_row_integrity
from flightrl.hardware.ranger_projection import points_from_rows, prepare_rows, trajectory_from_rows


SENSOR_COLORS = {
    "range.front": (31, 119, 180),
    "range.back": (255, 127, 14),
    "range.left": (44, 160, 44),
    "range.right": (214, 39, 40),
    "range.up": (148, 103, 189),
    "range.zrange": (140, 86, 75),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Crazyflie ranger room scan as PLY and interactive HTML")
    parser.add_argument("--input", required=True)
    parser.add_argument("--room-report", required=True)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--max-range-m", type=float, default=4.0)
    parser.add_argument("--min-drone-z-m", type=float, default=0.0)
    parser.add_argument("--raw-origin", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    prefix = Path(args.output_prefix) if args.output_prefix else input_path.with_suffix("")
    raw_rows = read_rows(input_path)
    integrity = ranger_row_integrity(raw_rows)
    if integrity["valid"] is not True:
        raise SystemExit(
            "room scan source integrity failed: "
            + ", ".join(integrity["failures"])
        )
    rows = prepare_rows(
        raw_rows,
        min_drone_z_m=args.min_drone_z_m,
        normalize_xy=not args.raw_origin,
    )
    points = points_from_rows(rows, max_range_m=args.max_range_m)
    trajectory = trajectory_from_rows(rows)
    if not points:
        raise SystemExit("no valid ranger points found")
    room = json.loads(Path(args.room_report).read_text())["room_estimate"]
    ply = prefix.with_suffix(".room.ply")
    html = prefix.with_suffix(".room.html")
    points_csv = prefix.with_suffix(".room_points.csv")
    write_points_csv(points_csv, points)
    write_ply(ply, points)
    write_html(html, points, trajectory, room)
    print(f"wrote {points_csv}")
    print(f"wrote {ply}")
    print(f"wrote {html}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_points_csv(path: Path, points) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "sensor", "x_m", "y_m", "z_m", "distance_m", "drone_x_m", "drone_y_m", "drone_z_m"))
        for point in points:
            writer.writerow((point.time_s, point.sensor, point.x_m, point.y_m, point.z_m, point.distance_m, point.drone_x_m, point.drone_y_m, point.drone_z_m))


def write_ply(path: Path, points) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point in points:
            red, green, blue = SENSOR_COLORS.get(point.sensor, (80, 80, 80))
            handle.write(f"{point.x_m:.6f} {point.y_m:.6f} {point.z_m:.6f} {red} {green} {blue}\n")


def write_html(path: Path, points, trajectory, room: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    for sensor in sorted({point.sensor for point in points}):
        subset = [point for point in points if point.sensor == sensor]
        red, green, blue = SENSOR_COLORS.get(sensor, (80, 80, 80))
        fig.add_trace(
            go.Scatter3d(
                x=[point.x_m for point in subset],
                y=[point.y_m for point in subset],
                z=[point.z_m for point in subset],
                mode="markers",
                name=sensor,
                marker={"size": 2, "color": f"rgb({red},{green},{blue})", "opacity": 0.55},
                hovertemplate=f"{sensor}<br>x=%{{x:.2f}} y=%{{y:.2f}} z=%{{z:.2f}}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=[pose.x_m for pose in trajectory],
            y=[pose.y_m for pose in trajectory],
            z=[pose.z_m for pose in trajectory],
            mode="lines",
            name="drone path",
            line={"color": "black", "width": 5},
        )
    )
    add_room_box(fig, room)
    fig.update_layout(
        title="Crazyflie Handheld Room Scan",
        scene={
            "xaxis_title": "x m",
            "yaxis_title": "y m",
            "zaxis_title": "z m",
            "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    fig.write_html(path, include_plotlyjs="cdn")


def add_room_box(fig: go.Figure, room: dict) -> None:
    corners = [
        (room["x_min"], room["y_min"], room["z_min"]),
        (room["x_max"], room["y_min"], room["z_min"]),
        (room["x_max"], room["y_max"], room["z_min"]),
        (room["x_min"], room["y_max"], room["z_min"]),
        (room["x_min"], room["y_min"], room["z_max"]),
        (room["x_max"], room["y_min"], room["z_max"]),
        (room["x_max"], room["y_max"], room["z_max"]),
        (room["x_min"], room["y_max"], room["z_max"]),
    ]
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    xs, ys, zs = [], [], []
    for start, end in edges:
        for index in (start, end):
            xs.append(corners[index][0])
            ys.append(corners[index][1])
            zs.append(corners[index][2])
        xs.append(None)
        ys.append(None)
        zs.append(None)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="estimated room box", line={"color": "rgba(0,0,0,0.45)", "width": 3}))


if __name__ == "__main__":
    main()
