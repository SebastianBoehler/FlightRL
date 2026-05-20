from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch

from flightrl.sixdof import SixDofPolicy


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_eval_accepts_task_subset(tmp_path: Path) -> None:
    checkpoint = tmp_path / "multitask.pt"
    torch.save(
        {
            "state_dict": SixDofPolicy(hidden_size=16, input_dim=30).state_dict(),
            "hidden_size": 16,
            "observation_dim": 30,
            "task": "position_yaw,obstacle_avoidance",
            "tasks": ["position_yaw", "obstacle_avoidance"],
        },
        checkpoint,
    )
    report = tmp_path / "subset.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_sixdof_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
            "--task",
            "obstacle_avoidance",
            "--steps",
            "4",
            "--num-envs",
            "4",
            "--output",
            str(report),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(report.read_text())
    assert data["tasks"] == ["obstacle_avoidance"]
    assert list(data["metrics"]["per_task"]) == ["obstacle_avoidance"]
    assert "mean_survival_fraction" in data["metrics"]
