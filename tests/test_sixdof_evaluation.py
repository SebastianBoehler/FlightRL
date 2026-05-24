from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from flightrl.sixdof import SixDofCrazyflieEnv, SixDofPolicy
from flightrl.sixdof.evaluation import position_error_for_task


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
    assert "mean_yaw_error_rad" in data["metrics"]


def test_checkpoint_eval_can_gate_yaw_error(tmp_path: Path) -> None:
    report = tmp_path / "teacher_yaw.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_sixdof_checkpoint.py"),
            "--teacher",
            "--task",
            "position_yaw",
            "--steps",
            "2",
            "--num-envs",
            "4",
            "--max-yaw-error-rad",
            "0.0",
            "--max-yaw-p95-error-rad",
            "0.0",
            "--output",
            str(report),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(report.read_text())
    assert data["thresholds"]["max_yaw_error_rad"] == 0.0
    assert data["thresholds"]["max_yaw_p95_error_rad"] == 0.0
    assert "yaw_error" in data["gate"]["failures"]
    assert "yaw_error_p95" in data["gate"]["failures"]


def test_circle_eval_position_error_uses_orbit_not_center() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=17, task="circle", reset_profile="circle_recovery")
    env.position[:] = np.asarray([[0.75, 0.0, 0.65]], dtype=np.float32)
    env.target_position[:] = np.asarray([[0.0, 0.0, 0.65]], dtype=np.float32)

    assert position_error_for_task(env, "circle")[0] < 1e-5
    assert position_error_for_task(env, "position_yaw")[0] > 0.7
