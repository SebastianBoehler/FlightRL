from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from sixdof_candidate_matrix_test_support import (
    MATRIX,
    ROOT,
    identity,
    record,
    suite_record,
    write_checkpoint,
)


def test_candidate_matrix_cli_ranks_and_reads_desktop_evidence(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint)
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"records": [suite_record("candidate", checkpoint)]}))
    model = tmp_path / "candidate.ts"
    model.write_bytes(b"desktop-model")
    parity = tmp_path / "parity.json"
    parity.write_text(
        json.dumps(
            {
                "schema": "flightrl.sixdof.desktop_export.v1",
                "evidence_scope": "desktop_cpu_only",
                "deployment_authority": False,
                "checkpoint": identity(checkpoint),
                "model": identity(model),
                "observation": {"mode": "base"},
                "parity": {"max_abs_error": 0.0},
            }
        )
    )
    latency = tmp_path / "latency.json"
    latency.write_text(
        json.dumps(
            {
                "schema": "flightrl.sixdof.desktop_latency.v1",
                "evidence_scope": "desktop_cpu_only",
                "deployment_authority": False,
                "checkpoint": identity(checkpoint),
                "torchscript": None,
                "eager": {
                    "per_sample_us": 3.0,
                    "samples_per_second": 333333.0,
                },
            }
        )
    )
    output = tmp_path / "matrix.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sixdof_candidate_matrix.py"),
            "--suite",
            str(suite),
            "--desktop-parity",
            f"candidate={parity}",
            "--desktop-latency",
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
    assert report["evidence_scope"] == "desktop_development"
    assert report["deployment_authority"] is False
    assert report["records"][0]["desktop_parity"]["passed"] is True
    assert report["records"][0]["controller"] == "policy"
    assert report["records"][0]["desktop_latency"]["per_sample_us"] == 3.0
    assert report["records"][0]["per_task_gate"]["position_yaw"]["passed"] is True
    assert report["records"][0]["checkpoint_meta"]["observation_mode"] == "base"
    assert report["records"][0]["mean_yaw_error_rad"] == 0.05
    assert report["best_by_task"]["position_yaw"]["yaw_error_p95_rad"] == 0.07
    assert report["best_by_task"]["position_yaw"]["controller"] == "policy"
    assert report["best_by_task"]["position_yaw"]["label"] == "candidate"
    assert output.with_suffix(".md").exists()


def test_candidate_matrix_prefers_passing_candidates() -> None:
    records = [
        {
            "label": "failed",
            "controller": "policy",
            "tasks": ["position_yaw"],
            "passed": False,
            "desktop_parity": {"present": True},
            "mean_completed_fraction": 0.9,
            "mean_survival_fraction": 0.9,
            "mean_position_error_m": 0.1,
            "clearance_p01_m": 0.2,
            "checkpoint": "a",
            "failures": ["completion"],
        },
        {
            "label": "passed",
            "controller": "policy",
            "tasks": ["position_yaw"],
            "passed": True,
            "desktop_parity": {"present": False},
            "mean_completed_fraction": 0.2,
            "mean_survival_fraction": 0.2,
            "mean_position_error_m": 2.0,
            "mean_yaw_error_rad": 0.1,
            "yaw_error_p95_rad": 0.2,
            "clearance_p01_m": 0.1,
            "checkpoint": "b",
            "failures": [],
        },
    ]
    assert MATRIX.best_by_task(records)["position_yaw"]["label"] == "passed"


def test_candidate_matrix_surfaces_best_multitask_candidate() -> None:
    records = [
        record("single", ["position_yaw"], completed=1.0, position_error=0.1),
        record(
            "multi_weak",
            ["position_yaw", "obstacle_avoidance"],
            completed=0.2,
            position_error=3.0,
        ),
        record(
            "multi_best",
            ["position_yaw", "obstacle_avoidance", "circle"],
            completed=0.8,
            position_error=0.5,
        ),
    ]

    best = MATRIX.best_multitask(records)

    assert best["label"] == "multi_best"
    assert best["tasks"] == ["position_yaw", "obstacle_avoidance", "circle"]


def test_candidate_matrix_score_uses_yaw_for_position_yaw() -> None:
    low_yaw = record(
        "low_yaw",
        ["position_yaw"],
        completed=0.8,
        position_error=1.0,
        yaw_error=0.1,
    )
    high_yaw = record(
        "high_yaw",
        ["position_yaw"],
        completed=0.8,
        position_error=1.0,
        yaw_error=0.8,
    )

    assert MATRIX.score(low_yaw) < MATRIX.score(high_yaw)


def test_candidate_matrix_keeps_teacher_residual_records(tmp_path: Path) -> None:
    checkpoint = tmp_path / "residual.pt"
    write_checkpoint(
        checkpoint,
        tasks=("circle",),
        controller="teacher_residual",
        residual_scale=0.05,
    )
    suite = tmp_path / "suite.json"
    record = suite_record("residual", checkpoint)
    record["controller"] = "teacher_residual"
    record["tasks"] = ["circle"]
    record["per_task_gate"] = {"circle": {"passed": True, "failures": []}}
    record["metrics"]["teacher_action_l2_mean"] = 0.001
    suite.write_text(json.dumps({"records": [record]}))

    records = MATRIX.checkpoint_records(suite, {}, {}, 1e-5)

    assert records[0]["controller"] == "teacher_residual"
    assert records[0]["checkpoint_meta"]["residual_scale"] == 0.05
    assert MATRIX.best_by_task(records)["circle"]["teacher_action_l2_mean"] == 0.001
