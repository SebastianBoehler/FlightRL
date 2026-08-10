from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from flightrl.hardware.ranger_integrity import ranger_row_integrity
from flightrl.hardware.ranger_map import estimate_room_bounds, summarize_map
from flightrl.hardware.ranger_projection import points_from_rows, prepare_rows, trajectory_from_rows


ROOT = Path(__file__).resolve().parents[1]


def test_summarize_map_marks_sufficient_scan_ready() -> None:
    rows = sample_rows(count=8, dt=1.5, x_step=0.06, yaw_step=12.0)
    points = points_from_rows(rows)
    trajectory = trajectory_from_rows(rows)

    summary = summarize_map(
        points,
        trajectory,
        min_points=24,
        min_duration_s=8.0,
        min_horizontal_sensors=4,
        min_trajectory_xy_span_m=0.25,
        min_yaw_span_deg=45.0,
        max_step_speed_m_s=1.0,
    )

    assert summary["mapping_ready"]
    assert summary["failures"] == []
    assert summary["trajectory"]["xy_span_m"] > 0.25
    assert summary["trajectory_quality"]["yaw_span_deg"] >= 80.0
    assert summary["trajectory_quality"]["mean_speed_m_s"] > 0.0
    assert summary["trajectory_quality"]["p95_speed_m_s"] > 0.0
    assert summary["trajectory_quality"]["speed_glitch_count"] == 0
    assert summary["point_density_per_path_m"] > 0.0
    assert summary["sensor_counts"]["range.front"] == len(rows)


def test_summarize_map_reports_static_or_sparse_scan_failures() -> None:
    rows = sample_rows(count=2, dt=0.2, x_step=0.0)
    summary = summarize_map(points_from_rows(rows), trajectory_from_rows(rows), min_yaw_span_deg=10.0)

    assert not summary["mapping_ready"]
    assert {"points", "duration", "trajectory_xy_span", "yaw_span"}.issubset(summary["failures"])


def test_summarize_map_reports_speed_glitches() -> None:
    rows = sample_rows(count=5, dt=1.0, x_step=0.1, yaw_step=12.0)
    rows[-1]["stateEstimate.x"] = "30.0"
    summary = summarize_map(points_from_rows(rows), trajectory_from_rows(rows), max_step_speed_m_s=5.0)

    assert not summary["mapping_ready"]
    assert "speed_glitch" in summary["failures"]
    assert summary["trajectory_quality"]["speed_glitch_count"] > 0


@pytest.mark.parametrize(("field", "value"), (("stateEstimate.x", "nan"), ("range.front", "inf")))
def test_room_summary_rejects_corrupt_source_rows(
    field: str,
    value: str,
) -> None:
    rows = sample_rows(count=8, dt=1.5, x_step=0.06, yaw_step=12.0)
    rows[3][field] = value
    integrity = ranger_row_integrity(rows)

    summary = summarize_map(
        points_from_rows(rows),
        trajectory_from_rows(rows),
        min_points=20,
        min_duration_s=8.0,
        min_horizontal_sensors=4,
        min_trajectory_xy_span_m=0.2,
        source_integrity=integrity,
    )

    assert integrity["valid"] is False
    assert summary["mapping_ready"] is False
    assert set(integrity["failures"]).issubset(summary["failures"])


def test_room_summary_rejects_nonmonotonic_trajectory() -> None:
    rows = sample_rows(count=8, dt=1.5, x_step=0.06, yaw_step=12.0)
    rows[4]["host_time_s"] = rows[3]["host_time_s"]

    summary = summarize_map(
        points_from_rows(rows),
        trajectory_from_rows(rows),
        source_integrity=ranger_row_integrity(rows),
    )

    assert summary["mapping_ready"] is False
    assert "trajectory_time_monotonic" in summary["failures"]


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


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    ((float("nan"), 4.0), (0.03, float("inf")), (4.0, 4.0), (5.0, 4.0)),
)
def test_ranger_projection_rejects_invalid_range_limits(
    minimum: float,
    maximum: float,
) -> None:
    with pytest.raises(ValueError, match="0 < min < max"):
        points_from_rows(
            sample_rows(count=2, dt=1.0, x_step=0.1),
            min_range_m=minimum,
            max_range_m=maximum,
        )


def test_estimate_room_bounds_uses_horizontal_points_and_floor_hits() -> None:
    rows = sample_rows(count=4, dt=1.0, x_step=0.1)
    estimate = estimate_room_bounds(points_from_rows(rows), trajectory_from_rows(rows), padding_m=0.05)

    assert estimate["x_min"] < -0.8
    assert estimate["x_max"] > 1.4
    assert estimate["y_min"] < -0.8
    assert estimate["y_max"] > 0.7
    assert estimate["z_min"] < 0.01
    assert estimate["down_point_count"] == len(rows)
    assert estimate["horizontal_point_count"] == len(rows) * 4


def test_estimate_room_bounds_clamps_noisy_floor_to_physical_floor() -> None:
    rows = sample_rows(count=4, dt=1.0, x_step=0.1)
    for row in rows:
        row["stateEstimate.z"] = "0.02"
        row["range.zrange"] = "90"

    estimate = estimate_room_bounds(points_from_rows(rows), trajectory_from_rows(rows), padding_m=0.05, floor_m=0.0)

    assert estimate["z_min"] == 0.0
    assert estimate["down_point_count"] == len(rows)


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
            "--min-yaw-span-deg",
            "0",
            "--max-step-speed-m-s",
            "5",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(output.read_text())
    assert data["summary"]["mapping_ready"]
    assert data["room_estimate"]["method"] == "axis_aligned_quantile_box"
    assert "trajectory_quality" in data["summary"]
    assert "point_density_per_path_m" in data["summary"]
    assert data["summary"]["trajectory_quality"]["speed_glitch_count"] == 0
    assert output.with_suffix(".md").exists()
    assert "mapping_ready=True" in result.stdout


def test_room_summary_cli_uses_device_time_for_bursted_flight_callbacks(
    tmp_path: Path,
) -> None:
    rows = sample_rows(count=4, dt=1.0, x_step=0.1)
    for index, row in enumerate(rows):
        row["crazyflie_time_ms"] = str(index * 1000)
    rows[2]["host_time_s"] = "1.001"
    log = tmp_path / "flight.csv"
    log.write_text(csv_text(rows))
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
            "--max-step-speed-m-s",
            "5",
            "--strict",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert data["preprocessing"]["time_source"] == "crazyflie_time_ms"
    assert data["summary"]["trajectory_quality"]["speed_glitch_count"] == 0


def sample_rows(*, count: int, dt: float, x_step: float, yaw_step: float = 0.0) -> list[dict[str, str]]:
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
                "stabilizer.yaw": str(index * yaw_step),
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
