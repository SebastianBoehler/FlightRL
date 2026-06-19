from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import torch

from flightrl.sixdof import SixDofPolicy


ROOT = Path(__file__).resolve().parents[1]


def test_sixdof_shadow_monitor_dry_run_is_monitor_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sixdof.pt"
    torch.save(
        {
            "state_dict": SixDofPolicy(hidden_size=16, input_dim=28).state_dict(),
            "hidden_size": 16,
            "observation_dim": 28,
            "tasks": ["obstacle_avoidance"],
            "task": "obstacle_avoidance",
            "controller": "teacher_residual",
            "residual_scale": 0.1,
        },
        checkpoint,
    )
    output = tmp_path / "shadow.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "crazyflie_sixdof_shadow_monitor.py"),
            "--checkpoint",
            str(checkpoint),
            "--task",
            "obstacle_avoidance",
            "--dry-run",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["monitor_only"] == "True"
    assert rows[0]["controls_drone"] == "False"
    assert "shadow_thrust" in rows[0]
