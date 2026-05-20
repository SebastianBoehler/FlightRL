from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from flightrl.replay import assess_log_quality


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    "host_time_s,stateEstimate.x,stateEstimate.y,stateEstimate.z,"
    "vx_m_s,vy_m_s,vz_m_s,yawrate_deg_s,range.front,range.back,range.left,range.right,range.up\n"
)


def test_assess_log_quality_accepts_calibration_ready_log() -> None:
    rows = [
        {
            "host_time_s": str(idx * 0.1),
            "stateEstimate.x": "0",
            "stateEstimate.y": "0",
            "stateEstimate.z": "0.4",
            "vx_m_s": "0",
            "vy_m_s": "0",
            "vz_m_s": "0",
            "yawrate_deg_s": "0",
            "range.front": "1000",
            "range.back": "1000",
            "range.left": "1000",
            "range.right": "1000",
            "range.up": "1000",
        }
        for idx in range(60)
    ]
    quality = assess_log_quality(rows, min_rows=50, min_duration_s=5.0)
    assert quality["calibration_ready"] is True
    assert quality["failures"] == []


def test_assess_log_quality_reports_missing_columns_and_bad_time() -> None:
    quality = assess_log_quality([{"host_time_s": "1", "range.front": "32766"}, {"host_time_s": "1", "range.front": "32766"}])
    assert quality["calibration_ready"] is False
    assert "missing_columns" in quality["failures"]
    assert "time_monotonic" in quality["failures"]
    assert "range_validity" in quality["failures"]


def test_replay_log_quality_cli_writes_report(tmp_path: Path) -> None:
    log = tmp_path / "log.csv"
    output = tmp_path / "quality.json"
    rows = "".join(f"{idx * 0.1},0,0,0.4,0,0,0,0,1000,1000,1000,1000,1000\n" for idx in range(60))
    log.write_text(HEADER + rows)
    subprocess.run(
        [
            sys.executable,
            "scripts/check_replay_log_quality.py",
            "--input",
            str(log),
            "--output",
            str(output),
            "--min-rows",
            "50",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["quality"]["calibration_ready"] is True
    assert output.with_suffix(".md").exists()
