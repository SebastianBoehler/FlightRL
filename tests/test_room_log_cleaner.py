from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from flightrl.hardware.room_log_cleaner import clean_room_rows


ROOT = Path(__file__).resolve().parents[1]


def test_clean_room_rows_drops_impossible_speed_spikes() -> None:
    rows = sample_rows()
    rows.insert(2, {**rows[1], "host_time_s": "2.0", "stateEstimate.x": "100.0"})

    result = clean_room_rows(rows, max_step_speed_m_s=5.0)

    assert result.input_count == 5
    assert result.kept_count == 4
    assert result.dropped_count == 1
    assert result.max_observed_step_speed_m_s > 50.0
    assert [float(row["stateEstimate.x"]) for row in result.rows] == [0.0, 0.2, 0.4, 0.6000000000000001]


def test_clean_room_rows_rejects_non_positive_threshold() -> None:
    try:
        clean_room_rows(sample_rows(), max_step_speed_m_s=0.0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_clean_room_log_cli_writes_clean_csv_and_report(tmp_path: Path) -> None:
    input_path = tmp_path / "room.csv"
    rows = sample_rows()
    rows.insert(2, {**rows[1], "host_time_s": "2.0", "stateEstimate.x": "100.0"})
    write_csv(input_path, rows)
    output = tmp_path / "room.clean.csv"
    report = tmp_path / "room.clean.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/clean_crazyflie_room_log.py",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--report",
            str(report),
            "--max-step-speed-m-s",
            "5",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    cleaned = list(csv.DictReader(output.open()))
    data = json.loads(report.read_text())
    assert len(cleaned) == 4
    assert data["dropped_count"] == 1
    assert "dropped=1/5" in result.stdout


def sample_rows() -> list[dict[str, str]]:
    rows = []
    for index in range(4):
        rows.append(
            {
                "host_time_s": str(index),
                "stateEstimate.x": str(index * 0.2),
                "stateEstimate.y": "0.0",
                "stateEstimate.z": "0.4",
                "stabilizer.roll": "0.0",
                "stabilizer.pitch": "0.0",
                "stabilizer.yaw": str(index * 5.0),
                "range.front": "1200",
                "range.back": "900",
                "range.left": "800",
                "range.right": "850",
                "range.up": "32766",
                "range.zrange": "400",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
