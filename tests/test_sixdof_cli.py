from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import torch

from flightrl.sixdof import SixDofPolicy, build_checkpoint_payload


ROOT = Path(__file__).resolve().parents[1]


def test_sixdof_training_and_rollout_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sixdof.pt"
    train = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_sixdof_teacher.py"),
            "--task",
            "position_yaw",
            "--updates",
            "1",
            "--steps-per-update",
            "2",
            "--num-envs",
            "8",
            "--batch-size",
            "16",
            "--eval-steps",
            "4",
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert checkpoint.exists()
    assert "checkpoint=" in train.stdout

    rollout = tmp_path / "rollout.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "rollout_sixdof_policy.py"),
            "--checkpoint",
            str(checkpoint),
            "--steps",
            "4",
            "--output",
            str(rollout),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    with rollout.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert "stateEstimate.x" in rows[0]


def test_sixdof_multitask_training_and_rollout_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sixdof_multitask.pt"
    train = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_sixdof_teacher.py"),
            "--task",
            "position_yaw,obstacle_avoidance",
            "--updates",
            "1",
            "--steps-per-update",
            "2",
            "--num-envs",
            "8",
            "--batch-size",
            "16",
            "--eval-steps",
            "4",
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert checkpoint.exists()
    assert "per_task" in train.stdout

    rollout = tmp_path / "multitask_rollout.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "rollout_sixdof_policy.py"),
            "--checkpoint",
            str(checkpoint),
            "--task",
            "obstacle_avoidance",
            "--steps",
            "4",
            "--output",
            str(rollout),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    with rollout.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert "action_thrust" in rows[0]

    report = tmp_path / "multitask_eval.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_sixdof_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
            "--steps",
            "4",
            "--num-envs",
            "8",
            "--output",
            str(report),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    evaluation = json.loads(report.read_text())
    assert "gate" in evaluation
    assert "teacher_action_l2_mean" in evaluation["metrics"]
    assert "action_saturation_fraction" in evaluation["metrics"]

    teacher_report = tmp_path / "teacher_eval.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_sixdof_checkpoint.py"),
            "--teacher",
            "--task",
            "position_yaw,obstacle_avoidance",
            "--steps",
            "4",
            "--num-envs",
            "8",
            "--output",
            str(teacher_report),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    teacher_evaluation = json.loads(teacher_report.read_text())
    assert teacher_evaluation["controller"] == "teacher"
    assert "teacher_action_l2_mean" not in teacher_evaluation["metrics"]
    assert "action_saturation_fraction" in teacher_evaluation["metrics"]


def test_history_observation_checkpoint_rollout_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sixdof_history.pt"
    torch.save(
        build_checkpoint_payload(
            state_dict=SixDofPolicy(hidden_size=16, input_dim=60).state_dict(),
            tasks=("position_yaw",),
            hidden_size=16,
            observation_mode="history1",
        ),
        checkpoint,
    )
    rollout = tmp_path / "history_rollout.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "rollout_sixdof_policy.py"), "--checkpoint", str(checkpoint), "--steps", "3", "--output", str(rollout)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    with rollout.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert "action_thrust" in rows[0]
