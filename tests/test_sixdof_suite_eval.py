from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import importlib.util

import torch

from flightrl.sixdof import SixDofPolicy


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_sixdof_suite", ROOT / "scripts" / "evaluate_sixdof_suite.py")
assert SPEC and SPEC.loader
SUITE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUITE
SPEC.loader.exec_module(SUITE)


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
    assert "mean_yaw_error_rad" in report["records"][0]["metrics"]
    assert "position_yaw" in report["records"][0]["per_task_gate"]
    best = report["summary"]["best_checkpoint_by_task"]["obstacle_avoidance"]
    assert best["label"] == "candidate"
    assert best["checkpoint"] == str(checkpoint)
    assert output.with_suffix(".md").exists()


def test_sixdof_suite_ranks_single_task_checkpoint_candidates() -> None:
    records = [
        record("weak", "position_yaw", False, 4.0, 0.4, 0.1),
        record("strong", "position_yaw", False, 2.0, 0.8, 0.2),
        record("passed", "position_yaw", True, 3.0, 0.7, 0.3),
        record("multi", "checkpoint", True, 0.1, 1.0, 1.0, tasks=["position_yaw", "circle"]),
    ]
    best = SUITE.best_checkpoint_by_task(records)
    assert best["position_yaw"]["label"] == "passed"
    assert "circle" not in best


def test_sixdof_suite_builds_per_task_gate_failures() -> None:
    metrics = {
        "per_task": {
            "position_yaw": {
                "clearance_p01_m": 0.05,
                "min_clearance_m": 0.05,
                "completed_fraction": 0.5,
                "survival_fraction": 0.5,
                "mean_position_error_m": 2.0,
                "mean_yaw_error_rad": 0.1,
                "yaw_error_p95_rad": 0.2,
            },
            "obstacle_avoidance": {
                "clearance_p01_m": 0.4,
                "min_clearance_m": 0.4,
                "completed_fraction": 1.0,
                "survival_fraction": 1.0,
                "mean_position_error_m": 0.2,
                "mean_yaw_error_rad": 0.0,
                "yaw_error_p95_rad": 0.0,
            },
        }
    }
    gates = SUITE.per_task_gate(metrics, {"min_clearance_m": 0.08, "min_completed_fraction": 0.9, "max_position_error_m": 1.0, "max_yaw_error_rad": None, "max_yaw_p95_error_rad": None})

    assert gates["position_yaw"]["failures"] == ["min_clearance", "completion", "position_error"]
    assert gates["obstacle_avoidance"]["passed"] is True


def record(label: str, task: str, passed: bool, position_error: float, completed: float, clearance: float, tasks=None) -> dict:
    return {
        "label": label,
        "controller": "checkpoint",
        "checkpoint": f"{label}.pt",
        "tasks": tasks or [task],
        "gate": {"passed": passed, "failures": [] if passed else ["position_error"]},
        "metrics": {
            "mean_position_error_m": position_error,
            "mean_yaw_error_rad": 0.2,
            "yaw_error_p95_rad": 0.3,
            "mean_completed_fraction": completed,
            "mean_survival_fraction": completed,
            "min_clearance_m": clearance,
            "clearance_p01_m": clearance,
        },
    }
