from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from flightrl.hardware.ranger_map import points_from_rows, prepare_rows, summarize_map, trajectory_from_rows


ROOT = Path(__file__).resolve().parents[1]


def test_summarize_map_marks_sufficient_scan_ready() -> None:
    rows = sample_rows(count=8, dt=1.5, x_step=0.06)
    points = points_from_rows(rows)
    trajectory = trajectory_from_rows(rows)

    summary = summarize_map(
        points,
        trajectory,
        min_points=24,
        min_duration_s=8.0,
        min_horizontal_sensors=4,
        min_trajectory_xy_span_m=0.25,
    )

    assert summary["mapping_ready"]
    assert summary["failures"] == []
    assert summary["trajectory"]["xy_span_m"] > 0.25
    assert summary["sensor_counts"]["range.front"] == len(rows)


def test_summarize_map_reports_static_or_sparse_scan_failures() -> None:
    rows = sample_rows(count=2, dt=0.2, x_step=0.0)
    summary = summarize_map(points_from_rows(rows), trajectory_from_rows(rows))

    assert not summary["mapping_ready"]
    assert {"points", "duration", "trajectory_xy_span"}.issubset(summary["failures"])


def test_prepare_rows_filters_height_and_normalizes_origin() -> None:
    rows = [
        {"host_time_s": "10", "stateEstimate.x": "4", "stateEstimate.y": "5", "stateEstimate.z": "0.1"},
        {"host_time_s": "11", "stateEstimate.x": "4.2", "stateEstimate.y": "5.3", "stateEstimate.z": "0.4"},
    ]

    prepared = prepare_rows(rows, min_drone_z_m=0.2, normalize_xy=True)

    assert len(prepared) == 1
    assert prepared[0]["host_time_s"] == "0.0"
    assert prepared[0]["stateEstimate.x"] == "0.0"
    assert prepared[0]["stateEstimate.y"] == "0.0"


def test_room_summary_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    log = tmp_path / "room.csv"
    log.write_text(csv_text(sample_rows(count=4, dt=1.0, x_step=0.1)))
    output = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_crazyflie_room.py",
            "--input",
            str(log),
            "--output",
            str(output),
            "--min-points",
            "12",
            "--min-duration-s",
            "2",
            "--min-trajectory-xy-span-m",
            "0.2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(output.read_text())
    assert data["summary"]["mapping_ready"]
    assert output.with_suffix(".md").exists()
    assert "mapping_ready=True" in result.stdout


def sample_rows(*, count: int, dt: float, x_step: float) -> list[dict[str, str]]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "host_time_s": str(index * dt),
                "stateEstimate.x": str(index * x_step),
                "stateEstimate.y": "0",
                "stateEstimate.z": "0.45",
                "stabilizer.roll": "0",
                "stabilizer.pitch": "0",
                "stabilizer.yaw": "0",
                "range.front": "1200",
                "range.back": "900",
                "range.left": "800",
                "range.right": "850",
                "range.up": "32766",
                "range.zrange": "450",
            }
        )
    return rows


def csv_text(rows: list[dict[str, str]]) -> str:
    header = list(rows[0])
    lines = [",".join(header)]
    lines.extend(",".join(row[key] for key in header) for row in rows)
    return "\n".join(lines) + "\n"
