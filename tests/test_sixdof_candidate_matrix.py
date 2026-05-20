from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import torch

from flightrl.sixdof import SixDofPolicy


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_sixdof_candidate_matrix", ROOT / "scripts" / "build_sixdof_candidate_matrix.py")
assert SPEC and SPEC.loader
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
SPEC.loader.exec_module(MATRIX)


def test_candidate_matrix_cli_ranks_and_reads_parity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "state_dict": SixDofPolicy(hidden_size=16).state_dict(),
            "hidden_size": 16,
            "observation_dim": 28,
            "observation_mode": "base",
        },
        checkpoint,
    )
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"records": [suite_record("candidate", checkpoint)]}))
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({"model": "candidate.ts", "observation": {"mode": "base"}, "parity": {"max_abs_error": 0.0}}))
    latency = tmp_path / "latency.json"
    latency.write_text(json.dumps({"eager": {"per_sample_us": 3.0, "samples_per_second": 333333.0}}))
    output = tmp_path / "matrix.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sixdof_candidate_matrix.py"),
            "--suite",
            str(suite),
            "--parity",
            f"candidate={parity}",
            "--latency",
            f"candidate={latency}",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text())
    assert report["records"][0]["edge_parity"]["passed"] is True
    assert report["records"][0]["edge_latency"]["per_sample_us"] == 3.0
    assert report["records"][0]["per_task_gate"]["position_yaw"]["passed"] is True
    assert report["records"][0]["checkpoint_meta"]["observation_mode"] == "base"
    assert report["records"][0]["mean_yaw_error_rad"] == 0.05
    assert report["best_by_task"]["position_yaw"]["yaw_error_p95_rad"] == 0.07
    assert report["best_by_task"]["position_yaw"]["label"] == "candidate"
    assert output.with_suffix(".md").exists()


def test_candidate_matrix_prefers_passing_candidates() -> None:
    records = [
        {"label": "failed", "tasks": ["position_yaw"], "passed": False, "edge_parity": {"present": True}, "mean_completed_fraction": 0.9, "mean_survival_fraction": 0.9, "mean_position_error_m": 0.1, "clearance_p01_m": 0.2, "checkpoint": "a", "failures": ["completion"]},
        {"label": "passed", "tasks": ["position_yaw"], "passed": True, "edge_parity": {"present": False}, "mean_completed_fraction": 0.2, "mean_survival_fraction": 0.2, "mean_position_error_m": 2.0, "mean_yaw_error_rad": 0.1, "yaw_error_p95_rad": 0.2, "clearance_p01_m": 0.1, "checkpoint": "b", "failures": []},
    ]
    assert MATRIX.best_by_task(records)["position_yaw"]["label"] == "passed"


def test_candidate_matrix_surfaces_best_multitask_candidate() -> None:
    records = [
        record("single", ["position_yaw"], completed=1.0, position_error=0.1),
        record("multi_weak", ["position_yaw", "obstacle_avoidance"], completed=0.2, position_error=3.0),
        record("multi_best", ["position_yaw", "obstacle_avoidance", "circle"], completed=0.8, position_error=0.5),
    ]

    best = MATRIX.best_multitask(records)

    assert best["label"] == "multi_best"
    assert best["tasks"] == ["position_yaw", "obstacle_avoidance", "circle"]


def test_candidate_matrix_score_uses_yaw_for_position_yaw() -> None:
    low_yaw = record("low_yaw", ["position_yaw"], completed=0.8, position_error=1.0, yaw_error=0.1)
    high_yaw = record("high_yaw", ["position_yaw"], completed=0.8, position_error=1.0, yaw_error=0.8)

    assert MATRIX.score(low_yaw) < MATRIX.score(high_yaw)


def suite_record(label: str, checkpoint: Path) -> dict:
    return {
        "label": label,
        "controller": "checkpoint",
        "checkpoint": str(checkpoint),
        "tasks": ["position_yaw"],
        "gate": {"passed": True, "failures": []},
        "per_task_gate": {"position_yaw": {"passed": True, "failures": []}},
        "metrics": {
            "mean_completed_fraction": 1.0,
            "mean_survival_fraction": 1.0,
            "mean_position_error_m": 0.1,
            "mean_yaw_error_rad": 0.05,
            "yaw_error_p95_rad": 0.07,
            "min_clearance_m": 0.2,
            "clearance_p01_m": 0.2,
        },
    }


def record(label: str, tasks: list[str], *, completed: float, position_error: float, yaw_error: float | None = None) -> dict:
    return {
        "label": label,
        "checkpoint": f"{label}.pt",
        "tasks": tasks,
        "passed": True,
        "failures": [],
        "mean_completed_fraction": completed,
        "mean_survival_fraction": completed,
        "mean_position_error_m": position_error,
        "mean_yaw_error_rad": yaw_error,
        "yaw_error_p95_rad": yaw_error,
        "clearance_p01_m": 0.2,
        "edge_parity": {"present": True, "passed": True},
        "edge_latency": {"present": True, "per_sample_us": 4.0},
    }
