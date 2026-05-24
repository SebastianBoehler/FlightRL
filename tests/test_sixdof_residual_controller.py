from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_residual_checkpoint_cli_creates_teacher_residual(tmp_path: Path) -> None:
    checkpoint = tmp_path / "residual.pt"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_sixdof_residual_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
            "--task",
            "circle",
            "--hidden-size",
            "16",
            "--residual-scale",
            "0.1",
            "--zero-weights",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    saved = torch.load(checkpoint, map_location="cpu")
    assert saved["controller"] == "teacher_residual"
    assert saved["residual_scale"] == 0.1
    assert saved["tasks"] == ["circle"]
    assert all(float(value.abs().max()) == 0.0 for value in saved["state_dict"].values())


def test_zero_residual_checkpoint_evaluates_like_teacher(tmp_path: Path) -> None:
    checkpoint = tmp_path / "residual.pt"
    report = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_sixdof_residual_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
            "--task",
            "circle",
            "--hidden-size",
            "16",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_sixdof_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
            "--task",
            "circle",
            "--steps",
            "20",
            "--num-envs",
            "16",
            "--reset-profile",
            "circle_recovery",
            "--output",
            str(report),
            "--max-yaw-error-rad",
            "0.6",
            "--max-yaw-p95-error-rad",
            "1.0",
            "--native-step",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(report.read_text())
    assert data["controller"] == "teacher_residual"
    assert data["metrics"]["teacher_action_l2_mean"] == 0.0
