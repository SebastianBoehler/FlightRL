from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from flightrl.sixdof.command_replay_sweep import ReplayCandidate, candidate_grid, sweep_command_replay


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_grid_builds_cartesian_product() -> None:
    candidates = candidate_grid(
        velocity_gains=[1.0, 2.0],
        yawrate_scales=[0.5],
        max_dt_values=[0.05, 0.08],
        override_z_m=0.5,
        hold_z_values=[None, 0.5],
    )

    assert len(candidates) == 8
    assert candidates[0].override_z_m == 0.5


def test_sweep_command_replay_sorts_by_score() -> None:
    records = sweep_command_replay(sample_rows(), room=None, candidates=[ReplayCandidate(1.0, 1.0, 0.05, 0.5, 0.5)])

    assert len(records) == 1
    assert records[0]["metrics"]["samples"] == 3.0
    assert records[0]["score"] >= 0.0


def test_sweep_command_replay_cli_writes_report(tmp_path: Path) -> None:
    input_path = tmp_path / "real.csv"
    output = tmp_path / "sweep.json"
    write_rows(input_path, sample_rows())

    result = subprocess.run(
        [
            sys.executable,
            "scripts/sweep_crazyflie_command_replay.py",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--override-z-m",
            "0.5",
            "--hold-z-values",
            "0.5",
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

    report = json.loads(output.read_text())
    assert report["best"]["params"]["hold_z_m"] == 0.5
    assert output.with_suffix(".md").exists()
    assert "candidates=1" in result.stdout


def sample_rows() -> list[dict[str, str]]:
    rows = []
    for index in range(3):
        rows.append(
            {
                "host_time_s": str(index * 0.05),
                "stateEstimate.x": str(index * 0.01),
                "stateEstimate.y": "0.0",
                "stateEstimate.z": "0.5",
                "stateEstimate.vx": "0.0",
                "stateEstimate.vy": "0.0",
                "stateEstimate.vz": "0.0",
                "stabilizer.roll": "0.0",
                "stabilizer.pitch": "0.0",
                "stabilizer.yaw": str(index * 1.0),
                "range.front": "1000",
                "range.back": "1000",
                "range.left": "1000",
                "range.right": "1000",
                "range.up": "1000",
                "range.zrange": "500",
                "vx_m_s": "0.1",
                "vy_m_s": "0.0",
                "vz_m_s": "0.0",
                "yawrate_deg_s": "10.0",
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
