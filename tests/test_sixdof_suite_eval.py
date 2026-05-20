from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch

from flightrl.sixdof import SixDofPolicy


ROOT = Path(__file__).resolve().parents[1]


def test_sixdof_suite_evaluates_teacher_and_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "state_dict": SixDofPolicy(hidden_size=16, input_dim=30).state_dict(),
            "hidden_size": 16,
            "observation_dim": 30,
            "tasks": ["position_yaw", "obstacle_avoidance"],
        },
        checkpoint,
    )
    output = tmp_path / "suite.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_sixdof_suite.py"),
            "--teacher",
            "teacher",
            "position_yaw",
            "--candidate",
            "candidate",
            str(checkpoint),
            "obstacle_avoidance",
            "--steps",
            "4",
            "--num-envs",
            "4",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["summary"]["total"] == 2
    assert report["records"][0]["controller"] == "teacher"
    assert report["records"][1]["tasks"] == ["obstacle_avoidance"]
    assert output.with_suffix(".md").exists()
