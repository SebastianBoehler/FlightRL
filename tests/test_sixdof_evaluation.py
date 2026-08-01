from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from flightrl.sixdof import SixDofCrazyflieEnv, SixDofPolicy, build_checkpoint_payload
from flightrl.sixdof.curriculum import ResetProfile
from flightrl.sixdof.evaluation import aggregate_task_metrics, evaluate_one, position_error_for_task
from flightrl.sixdof.gates import gate_status


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_eval_accepts_task_subset(tmp_path: Path) -> None:
    checkpoint = tmp_path / "multitask.pt"
    torch.save(
        build_checkpoint_payload(
            state_dict=SixDofPolicy(hidden_size=16, input_dim=30).state_dict(),
            tasks=("position_yaw", "obstacle_avoidance"),
            hidden_size=16,
        ),
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


def test_gate_can_reject_open_space_horizontal_speed() -> None:
    gate = gate_status(
        {
            "clearance_p01_m": 0.5,
            "min_clearance_m": 0.5,
            "mean_completed_fraction": 1.0,
            "mean_position_error_m": 0.1,
            "open_space_horizontal_speed_p95_m_s": 0.8,
        },
        min_clearance_m=0.08,
        min_completed_fraction=0.9,
        max_position_error_m=1.0,
        max_open_space_horizontal_speed_p95_m_s=0.45,
    )

    assert gate["passed"] is False
    assert "open_space_horizontal_speed_p95" in gate["failures"]


def test_gate_rejects_nonfinite_required_metric() -> None:
    gate = gate_status(
        {
            "clearance_p01_m": float("nan"),
            "mean_completed_fraction": 1.0,
            "mean_position_error_m": 0.1,
        },
        min_clearance_m=0.08,
        min_completed_fraction=0.9,
        max_position_error_m=1.0,
    )

    assert gate["passed"] is False
    assert "min_clearance_invalid" in gate["failures"]


def test_gate_rejects_nonfinite_threshold() -> None:
    gate = gate_status(
        {
            "clearance_p01_m": 0.5,
            "mean_completed_fraction": 1.0,
            "mean_position_error_m": 0.1,
        },
        min_clearance_m=float("nan"),
        min_completed_fraction=0.9,
        max_position_error_m=1.0,
    )

    assert gate == {"passed": False, "failures": ["thresholds_invalid"]}


def test_open_space_speed_excludes_envs_that_started_close() -> None:
    close_profile = ResetProfile(
        "unit_close",
        0.75,
        0.75,
        (0.5, 0.5),
        (0.5, 0.5),
        0.0,
        target_xy_offset_abs=0.0,
        target_z_offset_abs=0.0,
        target_yaw_offset_abs=0.0,
        near_wall_probability=1.0,
        near_wall_clearance_range=(0.1, 0.1),
    )

    metrics = aggregate_task_metrics(
        {
            "obstacle_avoidance": evaluate_one(
                lambda _model, env, *_args: np.zeros((env.num_envs, 4), dtype=np.float32),
                None,
                ("obstacle_avoidance",),
                "obstacle_avoidance",
                123,
                3,
                8,
                False,
                close_profile,
                None,
                "base",
            )
        }
    )

    assert metrics["open_space_horizontal_speed_p95_m_s"] == 0.0
