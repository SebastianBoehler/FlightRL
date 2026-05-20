from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from flightrl.hardware.ranger_map import estimate_room_bounds, points_from_rows, prepare_rows, summarize_map, trajectory_from_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Crazyflie ranger logs for room-map quality")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--max-range-m", type=float, default=4.0)
    parser.add_argument("--min-range-m", type=float, default=0.03)
    parser.add_argument("--min-drone-z-m", type=float, default=0.0)
    parser.add_argument("--raw-origin", action="store_true")
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--min-duration-s", type=float, default=10.0)
    parser.add_argument("--min-horizontal-sensors", type=int, default=3)
    parser.add_argument("--min-trajectory-xy-span-m", type=float, default=0.25)
    parser.add_argument("--room-padding-m", type=float, default=0.05)
    parser.add_argument("--strict", action="store_true", help="exit non-zero when mapping_ready is false")
    args = parser.parse_args()

    input_path = Path(args.input)
    output = Path(args.output or input_path.with_suffix(".room.json"))
    markdown = Path(args.markdown or output.with_suffix(".md"))
    rows = load_rows(input_path, min_drone_z_m=args.min_drone_z_m, normalize_xy=not args.raw_origin)
    points = points_from_rows(rows, max_range_m=args.max_range_m, min_range_m=args.min_range_m)
    trajectory = trajectory_from_rows(rows)
    summary = summarize_map(
        points,
        trajectory,
        min_points=args.min_points,
        min_duration_s=args.min_duration_s,
        min_horizontal_sensors=args.min_horizontal_sensors,
        min_trajectory_xy_span_m=args.min_trajectory_xy_span_m,
    )
    report = {
        "input": str(input_path),
        "preprocessing": {
            "max_range_m": args.max_range_m,
            "min_range_m": args.min_range_m,
            "min_drone_z_m": args.min_drone_z_m,
            "normalize_xy": not args.raw_origin,
        },
        "thresholds": {
            "min_points": args.min_points,
            "min_duration_s": args.min_duration_s,
            "min_horizontal_sensors": args.min_horizontal_sensors,
            "min_trajectory_xy_span_m": args.min_trajectory_xy_span_m,
        },
        "summary": summary,
        "room_estimate": estimate_room_bounds(points, trajectory, padding_m=args.room_padding_m, max_range_m=args.max_range_m),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    markdown.write_text(render_markdown(report))
    print(f"wrote {output} and {markdown}; mapping_ready={summary['mapping_ready']}")
    if args.strict and not summary["mapping_ready"]:
        raise SystemExit(1)


def load_rows(path: Path, *, min_drone_z_m: float, normalize_xy: bool) -> list[dict[str, str | float]]:
    with path.open() as handle:
        return prepare_rows(csv.DictReader(handle), min_drone_z_m=min_drone_z_m, normalize_xy=normalize_xy)


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    failures = ", ".join(summary["failures"]) or "none"
    active = ", ".join(summary["active_horizontal_sensors"]) or "none"
    rows = [
        ("input", report["input"]),
        ("mapping_ready", str(summary["mapping_ready"])),
        ("failures", failures),
        ("point_count", str(summary["point_count"])),
        ("pose_count", str(summary["pose_count"])),
        ("duration_s", f"{summary['duration_s']:.2f}"),
        ("points_per_second", f"{summary['points_per_second']:.1f}"),
        ("active_horizontal_sensors", active),
        ("trajectory_xy_span_m", f"{summary['trajectory']['xy_span_m']:.3f}"),
        ("trajectory_path_length_m", f"{summary['trajectory_path_length_m']:.3f}"),
        ("point_cloud_xy_span_m", f"{summary['point_cloud']['xy_span_m']:.3f}"),
        ("point_cloud_z_span_m", f"{summary['point_cloud']['z_span_m']:.3f}"),
    ]
    table = ["| metric | value |", "| --- | --- |"]
    table.extend(f"| {key} | {value} |" for key, value in rows)
    sensor_rows = ["| sensor | points |", "| --- | ---: |"]
    sensor_rows.extend(f"| {sensor} | {count} |" for sensor, count in summary["sensor_counts"].items())
    room = report["room_estimate"]
    room_rows = ["| bound | value m |", "| --- | ---: |"]
    room_rows.extend(
        f"| {key} | {room[key]:.3f} |"
        for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "width_m", "depth_m", "height_m")
    )
    warnings = ", ".join(room["warnings"]) or "none"
    return "\n".join(
        [
            "# Crazyflie Room Map Quality",
            "",
            *table,
            "",
            "## Sensor Coverage",
            "",
            *sensor_rows,
            "",
            "## Estimated Box Room",
            "",
            *room_rows,
            "",
            f"Warnings: {warnings}",
            "",
        ]
    ) + "\n"


if __name__ == "__main__":
    main()
