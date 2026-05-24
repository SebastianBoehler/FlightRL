from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from flightrl.hardware.calibration_flight import build_calibration_sequence, sequence_duration_s
from flightrl.hardware.calibration_quality import summarize_calibration_log


ROOT = Path(__file__).resolve().parents[1]


def test_build_calibration_sequence_contains_replay_modes() -> None:
    sequence = build_calibration_sequence(segment_s=1.0, hover_s=0.5, speed_m_s=0.1, yawrate_deg_s=15.0)
    modes = [command.mode for command in sequence]

    assert modes[0] == "hover_start"
    assert "line_x_pos" in modes
    assert "line_y_neg" in modes
    assert "yaw_pos" in modes
    assert "square_y_neg" in modes
    assert sequence_duration_s(sequence) == 11.0


def test_calibration_summary_marks_complete_log_ready() -> None:
    rows = sample_rows()

    summary = summarize_calibration_log(rows, min_rows=10, min_duration_s=1.0, min_floor_valid_ratio=0.9, min_yaw_span_deg=20.0)

    assert summary["replay_calibration_ready"]
    assert summary["failures"] == []
    assert summary["command_axes"]["vx_pos"]
    assert summary["floor_valid_ratio"] == 1.0


def test_calibration_summary_rejects_missing_floor_and_modes() -> None:
    rows = [{"host_time_s": "0", "range.zrange": "32766", "stabilizer.yaw": "0", "stateEstimate.x": "0", "stateEstimate.y": "0"}]

    summary = summarize_calibration_log(rows, min_rows=2, min_duration_s=1.0)

    assert not summary["replay_calibration_ready"]
    assert {"rows", "duration", "missing_columns", "floor_range", "yaw_span", "command_modes"}.issubset(summary["failures"])
    assert "vx_m_s" in summary["missing_columns"]


def test_calibration_summary_rejects_non_monotonic_time() -> None:
    rows = sample_rows()
    rows[2]["host_time_s"] = rows[1]["host_time_s"]

    summary = summarize_calibration_log(rows, min_rows=10, min_duration_s=1.0, min_floor_valid_ratio=0.9, min_yaw_span_deg=20.0)

    assert not summary["replay_calibration_ready"]
    assert "time_monotonic" in summary["failures"]


def test_calibration_flight_dry_run_runs_without_cflib() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/crazyflie_calibration_flight.py", "--dry-run", "--segment-s", "0.2"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "dry_run calibration sequence" in result.stdout
    assert "line_x_pos" in result.stdout


def test_calibration_summary_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    log = tmp_path / "calibration.csv"
    write_rows(log, sample_rows())
    output = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_crazyflie_calibration.py",
            "--input",
            str(log),
            "--output",
            str(output),
            "--min-rows",
            "10",
            "--min-duration-s",
            "1",
            "--min-floor-valid-ratio",
            "0.9",
            "--min-yaw-span-deg",
            "20",
            "--strict",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(output.read_text())
    assert data["summary"]["replay_calibration_ready"]
    assert output.with_suffix(".md").exists()
    assert "replay_calibration_ready=True" in result.stdout


def sample_rows() -> list[dict[str, str]]:
    modes = ["line_x_pos", "line_x_neg", "line_y_pos", "line_y_neg", "yaw_pos", "yaw_neg"]
    rows = []
    for index in range(30):
        mode = modes[index % len(modes)]
        rows.append(
            {
                "host_time_s": str(index * 0.1),
                "mode": mode,
                "vx_m_s": "0.1" if mode == "line_x_pos" else "-0.1" if mode == "line_x_neg" else "0.0",
                "vy_m_s": "0.1" if mode == "line_y_pos" else "-0.1" if mode == "line_y_neg" else "0.0",
                "vz_m_s": "0.0",
                "yawrate_deg_s": "20.0" if mode == "yaw_pos" else "-20.0" if mode == "yaw_neg" else "0.0",
                "range.zrange": "550",
                "range.front": "1500",
                "range.back": "1500",
                "range.left": "1200",
                "range.right": "1200",
                "range.up": "1800",
                "stabilizer.yaw": str(index * 2.0),
                "stateEstimate.x": str(index * 0.01),
                "stateEstimate.y": str(index * 0.005),
                "stateEstimate.z": "0.55",
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
