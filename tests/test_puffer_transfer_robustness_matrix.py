from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy


ROOT = Path(__file__).resolve().parents[1]


def test_robustness_matrix_cli_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.bin"
    log = tmp_path / "log.csv"
    output = tmp_path / "matrix.json"
    torch.save(PufferSixDofPolicy(PufferPolicyMetadata(28, 16, 4, 1)).state_dict(), checkpoint)
    write_log(log)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_puffer_transfer_robustness_matrix.py"),
            "--label",
            "tiny_matrix",
            "--obstacle-checkpoint",
            str(checkpoint),
            "--velocity-checkpoint",
            str(checkpoint),
            "--obstacle-live-log",
            f"smoke:{log}",
            "--velocity-live-log",
            f"vel:{log}",
            "--seed",
            "11",
            "--seed",
            "22",
            "--steps",
            "2",
            "--num-envs",
            "4",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "puffer_transfer_robustness_matrix=" in result.stdout
    report = json.loads(output.read_text())
    assert report["label"] == "tiny_matrix"
    assert report["seeds"] == [11, 22]
    assert [run["seed"] for run in report["runs"]] == [11, 22]
    assert output.with_suffix(".md").exists()


def write_log(path: Path) -> None:
    rows = [telemetry_row() for _ in range(4)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def telemetry_row() -> dict[str, float]:
    return {
        "stateEstimate.x": 0.0,
        "stateEstimate.y": 0.0,
        "stateEstimate.z": 0.5,
        "stateEstimate.vx": 0.0,
        "stateEstimate.vy": 0.0,
        "stateEstimate.vz": 0.0,
        "stabilizer.roll": 0.0,
        "stabilizer.pitch": 0.0,
        "stabilizer.yaw": 0.0,
        "gyro.x": 0.0,
        "gyro.y": 0.0,
        "gyro.z": 0.0,
        "range.front": 800.0,
        "range.back": 900.0,
        "range.left": 700.0,
        "range.right": 600.0,
        "range.up": 1500.0,
        "range.zrange": 500.0,
        "sys.canfly": 1.0,
        "sys.isTumbled": 0.0,
        "vx_m_s": 0.0,
        "vy_m_s": 0.0,
        "vz_m_s": 0.0,
        "yawrate_deg_s": 0.0,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.5,
    }
