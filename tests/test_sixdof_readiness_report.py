from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from sixdof_readiness_test_support import (
    READINESS,
    ROOT,
    candidate_record,
    native_report,
    puffer_export_report,
    profile_matrix,
    replay_report,
    residual_sweep_report,
    room_report,
    throughput_report,
)


def test_readiness_report_cli_promotes_complete_candidate(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    room = tmp_path / "room.json"
    native = tmp_path / "native.json"
    replay = tmp_path / "replay.json"
    residual = tmp_path / "residual.json"
    throughput = tmp_path / "throughput.json"
    puffer = tmp_path / "puffer.json"
    output = tmp_path / "readiness.json"
    candidate_checkpoint = tmp_path / "candidate.pt"
    multitask_checkpoint = tmp_path / "multi.pt"
    candidate_checkpoint.write_bytes(b"checkpoint")
    multitask_checkpoint.write_bytes(b"checkpoint")
    candidate_model = tmp_path / "candidate.ts"
    multitask_model = tmp_path / "multi.ts"
    candidate_model.write_bytes(b"model")
    multitask_model.write_bytes(b"model")
    matrix.write_text(
        json.dumps(
            {
                "evidence_scope": "desktop_development",
                "deployment_authority": False,
                "best_by_task": {
                    "obstacle_avoidance": candidate_record(
                        checkpoint=str(candidate_checkpoint), model=candidate_model
                    )
                },
                "best_multitask": candidate_record(
                    label="multi",
                    checkpoint=str(multitask_checkpoint),
                    model=multitask_model,
                    tasks=["position_yaw", "obstacle_avoidance"],
                    latency=None,
                ),
            }
        )
    )
    room.write_text(json.dumps(room_report(mapping_ready=True)))
    native.write_text(json.dumps(native_report(state_rmse=1e-8, range_rmse=0.1)))
    replay.write_text(json.dumps(replay_report(state_rmse=0.05, range_rmse=120.0)))
    residual.write_text(json.dumps(residual_sweep_report()))
    throughput.write_text(json.dumps(throughput_report()))
    puffer.write_text(json.dumps(puffer_export_report()))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sixdof_readiness_report.py"),
            "--matrix",
            str(matrix),
            "--room-report",
            str(room),
            "--native-parity",
            str(native),
            "--replay-comparison",
            str(replay),
            "--residual-sweep",
            str(residual),
            "--training-throughput",
            str(throughput),
            "--puffer-export",
            str(puffer),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text())
    assert report["records"][0]["ready"] is True
    assert report["records"][0]["checkpoint_exists"] is True
    assert report["global_evidence"]["replay_comparison"]["passed"] is True
    assert report["global_evidence"]["residual_sweep"]["best"]["name"] == "scale005"
    assert (
        report["global_evidence"]["training_throughput"]["best_total_sps"][
            "total_sps"
        ]
        == 123456.0
    )
    assert report["global_evidence"]["puffer_export"]["passed"] is True
    assert report["records"][0]["sim"]["mean_yaw_error_rad"] == 0.05
    assert report["records"][0]["sim"]["yaw_error_p95_rad"] == 0.07
    assert (
        report["records"][0]["sim"]["per_task_gate"]["obstacle_avoidance"][
            "passed"
        ]
        is True
    )
    assert report["summary"]["ready_tasks"] == ["obstacle_avoidance"]
    assert report["records"][1]["task"] == "multitask"
    assert report["records"][1]["tasks"] == [
        "position_yaw",
        "obstacle_avoidance",
    ]
    assert report["evidence_scope"] == "desktop_development"
    assert report["deployment_authority"] is False
    assert "desktop_latency_missing" in report["records"][1]["failures"]
    assert output.with_suffix(".md").exists()


def test_readiness_report_surfaces_missing_evidence() -> None:
    evidence = {
        "room": {"present": False, "mapping_ready": False},
        "native_parity": {"present": False, "passed": False},
        "replay_comparison": {"present": False, "required": True, "passed": False},
    }
    record = READINESS.evaluate_record(
        (
            "position_yaw",
            candidate_record(
                passed=False,
                parity=False,
                latency=None,
                tasks=["position_yaw"],
            ),
        ),
        evidence,
        50.0,
    )

    assert record["ready"] is False
    assert {
        "sim_gate",
        "desktop_parity",
        "desktop_latency_missing",
        "room_map",
        "native_parity",
        "replay_comparison_missing",
    }.issubset(record["failures"])


def test_readiness_report_blocks_missing_checkpoint_when_required() -> None:
    evidence = {
        "room": {"present": True, "mapping_ready": True},
        "native_parity": {"present": True, "passed": True},
        "replay_comparison": {"present": False, "required": False, "passed": True},
    }

    record = READINESS.evaluate_record(
        ("obstacle_avoidance", candidate_record(checkpoint="missing.pt")),
        evidence,
        50.0,
        require_checkpoint_file=True,
    )

    assert record["ready"] is False
    assert record["checkpoint_exists"] is False
    assert "checkpoint_missing" in record["failures"]


def test_readiness_report_blocks_failed_profile_matrix() -> None:
    evidence = {
        "room": {"present": True, "mapping_ready": True},
        "native_parity": {"present": True, "passed": True},
        "replay_comparison": {"present": False, "required": False, "passed": True},
        "profile_matrix": READINESS.compact_profile_matrix(
            profile_matrix(passed=False)
        ),
    }
    record = READINESS.evaluate_record(
        ("position_yaw", candidate_record(tasks=["position_yaw"])), evidence, 50.0
    )

    assert record["ready"] is False
    assert "profile_matrix" in record["failures"]
    assert record["profile_matrix"]["worst_survival_fraction"] == 0.7


def test_readiness_report_requires_position_yaw_profile_evidence_when_matrix_present() -> None:
    evidence = {
        "room": {"present": True, "mapping_ready": True},
        "native_parity": {"present": True, "passed": True},
        "replay_comparison": {"present": False, "required": False, "passed": True},
        "profile_matrix": READINESS.compact_profile_matrix(
            {"profiles": ["broad"], "records": []}
        ),
    }
    record = READINESS.evaluate_record(
        ("position_yaw", candidate_record(tasks=["position_yaw"])), evidence, 50.0
    )

    assert "profile_matrix_missing" in record["failures"]
