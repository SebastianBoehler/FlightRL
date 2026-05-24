from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_calibration_replay_pipeline_blocks_unready_quality(tmp_path: Path) -> None:
    log = tmp_path / "bad.csv"
    write_rows(log, [sample_row(0, "hover_start")])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_calibration_replay_report.py",
            "--input",
            str(log),
            "--room-report",
            str(room_report(tmp_path)),
            "--matrix",
            "artifacts/replay/sixdof_candidate_matrix_current.json",
            "--native-parity",
            "artifacts/replay/sixdof_native_parity_current.json",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "calibration quality failed" in result.stderr
    assert (tmp_path / "out" / "bad.quality.json").exists()


def test_calibration_replay_pipeline_writes_outputs_when_quality_ready(tmp_path: Path) -> None:
    log = tmp_path / "calibration.csv"
    write_rows(log, sample_rows())
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_calibration_replay_report.py",
            "--input",
            str(log),
            "--room-report",
            str(room_report(tmp_path)),
            "--matrix",
            "artifacts/replay/sixdof_candidate_matrix_current.json",
            "--native-parity",
            "artifacts/replay/sixdof_native_parity_current.json",
            "--profile-matrix",
            str(profile_matrix(tmp_path)),
            "--output-dir",
            str(output_dir),
            "--prefix",
            "cal",
            "--override-z-m",
            "0.55",
            "--hold-z-values",
            "0.55",
            "--velocity-gains",
            "1.0",
            "--yawrate-scales",
            "1.0",
            "--max-dt-values",
            "0.05",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "readiness=" in result.stdout
    assert json.loads((output_dir / "cal.quality.json").read_text())["summary"]["replay_calibration_ready"]
    assert json.loads((output_dir / "cal.readiness.json").read_text())["profile_matrix"] == str(tmp_path / "profile.json")
    assert (output_dir / "cal.sweep.json").exists()
    assert (output_dir / "cal.comparison.json").exists()
    assert (output_dir / "cal.readiness.md").exists()


def sample_rows() -> list[dict[str, str]]:
    modes = ["line_x_pos", "line_x_neg", "line_y_pos", "line_y_neg", "yaw_pos", "yaw_neg"]
    return [sample_row(index, modes[index % len(modes)]) for index in range(110)]


def sample_row(index: int, mode: str) -> dict[str, str]:
    return {
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
        "stabilizer.roll": "0.0",
        "stabilizer.pitch": "0.0",
        "stabilizer.yaw": str(index),
        "stateEstimate.x": str(index * 0.002),
        "stateEstimate.y": str(index * 0.001),
        "stateEstimate.z": "0.55",
        "stateEstimate.vx": "0.0",
        "stateEstimate.vy": "0.0",
        "stateEstimate.vz": "0.0",
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def room_report(tmp_path: Path) -> Path:
    path = tmp_path / "room.json"
    path.write_text(
        json.dumps(
            {
                "summary": {"mapping_ready": True, "failures": [], "point_count": 100, "duration_s": 10.0},
                "room_estimate": {
                    "x_min": -2.0,
                    "x_max": 2.0,
                    "y_min": -2.0,
                    "y_max": 2.0,
                    "z_min": 0.0,
                    "z_max": 2.5,
                    "max_range_m": 4.0,
                },
            }
        )
        + "\n"
    )
    return path


def profile_matrix(tmp_path: Path) -> Path:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"profiles": ["broad"], "records": []}) + "\n")
    return path
