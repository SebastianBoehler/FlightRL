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


def record(label: str, task: str, passed: bool, position_error: float, completed: float, clearance: float, tasks=None) -> dict:
    return {
        "label": label,
        "controller": "checkpoint",
        "checkpoint": f"{label}.pt",
        "tasks": tasks or [task],
        "gate": {"passed": passed, "failures": [] if passed else ["position_error"]},
        "metrics": {
            "mean_position_error_m": position_error,
            "mean_completed_fraction": completed,
            "mean_survival_fraction": completed,
            "min_clearance_m": clearance,
            "clearance_p01_m": clearance,
        },
    }
