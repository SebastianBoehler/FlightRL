from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch

from flightrl.sixdof import SixDofPolicy, build_checkpoint_payload
from flightrl.sixdof.diagnostics import diagnose_controller, summarize_diagnostics


ROOT = Path(__file__).resolve().parents[1]


def test_teacher_diagnostics_returns_timeline_bins() -> None:
    report = diagnose_controller(
        None,
        ("position_yaw",),
        task="position_yaw",
        reset_profile="position_yaw_easy",
        seed=3,
        steps=12,
        num_envs=4,
        bins=3,
    )

    assert report["final"]["survival_fraction"] >= 0.0
    assert len(report["timeline"]) == 3
    assert report["timeline"][0]["step_start"] == 0.0


def test_summarize_diagnostics_reports_position_error_blocker() -> None:
    summary = summarize_diagnostics(
        [
            {
                "task": "position_yaw",
                "reset_profile": "wide",
                "final": {"survival_fraction": 1.0, "position_error_mean_m": 2.0, "clearance_p01_m": 0.5},
            }
        ]
    )

    assert summary["blocked_count"] == 1
    assert summary["blocked"][0]["reason"] == "position_error"


def test_diagnose_sixdof_policy_teacher_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "diagnostics.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_sixdof_policy.py",
            "--teacher",
            "--task",
            "position_yaw",
            "--profiles",
            "position_yaw_easy",
            "--steps",
            "8",
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

    data = json.loads(output.read_text())
    assert data["controller"] == "teacher"
    assert data["records"][0]["reset_profile"] == "position_yaw_easy"
    assert output.with_suffix(".md").exists()
    assert "diagnostics=" in result.stdout


def test_diagnose_sixdof_policy_supports_teacher_residual_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "residual.pt"
    output = tmp_path / "diagnostics.json"
    torch.save(
        build_checkpoint_payload(
            state_dict=SixDofPolicy(hidden_size=16).state_dict(),
            tasks=("position_yaw",),
            hidden_size=16,
            controller="teacher_residual",
        ),
        checkpoint,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_sixdof_policy.py",
            "--checkpoint",
            str(checkpoint),
            "--task",
            "position_yaw",
            "--profiles",
            "position_yaw_easy",
            "--steps",
            "8",
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

    data = json.loads(output.read_text())
    assert data["controller"] == "teacher_residual"
    assert "yaw_error_p95_rad" in data["records"][0]["final"]
    assert "settled_half" in data["records"][0]["phase_summary"]
