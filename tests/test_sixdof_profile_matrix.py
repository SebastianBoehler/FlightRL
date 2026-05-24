from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_sixdof_profile_matrix", ROOT / "scripts" / "build_sixdof_profile_matrix.py")
assert SPEC and SPEC.loader
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
SPEC.loader.exec_module(MATRIX)


def test_profile_matrix_aggregates_worst_profile_metrics(tmp_path: Path) -> None:
    easy = write_suite(tmp_path / "easy.json", "position_yaw_easy", completed=1.0, survival=1.0, position_error=0.4, passed=True)
    recovery = write_suite(
        tmp_path / "recovery.json",
        "position_yaw_recovery",
        completed=0.2,
        survival=0.3,
        position_error=5.0,
        passed=False,
    )

    report = MATRIX.build_profile_matrix([easy, recovery])
    record = report["records"][0]

    assert report["profiles"] == ["position_yaw_easy", "position_yaw_recovery"]
    assert record["passed_all_profiles"] is False
    assert record["worst_survival_fraction"] == 0.3
    assert record["worst_position_error_m"] == 5.0
    assert record["failures_by_profile"]["position_yaw_recovery"] == ["completion"]
    assert report["task_records"][0]["task"] == "position_yaw"
    assert report["task_records"][0]["worst_yaw_p95_rad"] == 1.2
    assert report["best_by_task"]["position_yaw"]["label"] == "candidate"


def test_profile_matrix_cli_writes_markdown(tmp_path: Path) -> None:
    suite = write_suite(tmp_path / "suite.json", "broad", completed=0.9, survival=0.9, position_error=0.8, passed=True)
    output = tmp_path / "matrix.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sixdof_profile_matrix.py"),
            "--suite",
            str(suite),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text())
    assert report["records"][0]["passed_all_profiles"] is True
    assert report["task_records"][0]["passed_all_profiles"] is True
    assert output.with_suffix(".md").exists()
    assert "Per-Task Blockers" in output.with_suffix(".md").read_text()


def test_profile_matrix_keeps_teacher_residual_candidates(tmp_path: Path) -> None:
    suite = write_suite(tmp_path / "suite.json", "broad", completed=0.9, survival=0.9, position_error=0.8, passed=True, controller="teacher_residual")

    report = MATRIX.build_profile_matrix([suite])

    assert report["records"][0]["controller"] == "teacher_residual"
    assert report["best_by_task"]["position_yaw"]["controller"] == "teacher_residual"


def write_suite(path: Path, profile: str, *, completed: float, survival: float, position_error: float, passed: bool, controller: str = "checkpoint") -> Path:
    path.write_text(
        json.dumps(
            {
                "reset_profile": profile,
                "records": [
                    {
                        "label": "candidate",
                        "controller": controller,
                        "checkpoint": "candidate.pt",
                        "tasks": ["position_yaw"],
                        "gate": {"passed": passed, "failures": [] if passed else ["completion"]},
                        "per_task_gate": {
                            "position_yaw": {"passed": passed, "failures": [] if passed else ["completion", "yaw_error_p95"]}
                        },
                        "metrics": {
                            "mean_completed_fraction": completed,
                            "mean_survival_fraction": survival,
                            "mean_position_error_m": position_error,
                            "mean_yaw_error_rad": 0.1,
                            "yaw_error_p95_rad": 0.2,
                            "min_clearance_m": 0.1,
                            "clearance_p01_m": 0.1,
                            "per_task": {
                                "position_yaw": {
                                    "completed_fraction": completed,
                                    "survival_fraction": survival,
                                    "mean_position_error_m": position_error,
                                    "mean_yaw_error_rad": 0.4 if not passed else 0.1,
                                    "yaw_error_p95_rad": 1.2 if not passed else 0.2,
                                    "min_clearance_m": 0.1,
                                    "clearance_p01_m": 0.1,
                                }
                            },
                        },
                    }
                ],
            }
        )
    )
    return path
