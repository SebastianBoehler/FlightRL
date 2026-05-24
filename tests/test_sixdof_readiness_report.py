from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_sixdof_readiness_report", ROOT / "scripts" / "build_sixdof_readiness_report.py")
assert SPEC and SPEC.loader
READINESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = READINESS
SPEC.loader.exec_module(READINESS)


def test_readiness_report_cli_promotes_complete_candidate(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    room = tmp_path / "room.json"
    native = tmp_path / "native.json"
    replay = tmp_path / "replay.json"
    residual = tmp_path / "residual.json"
    throughput = tmp_path / "throughput.json"
    puffer = tmp_path / "puffer.json"
    output = tmp_path / "readiness.json"
    matrix.write_text(
        json.dumps(
            {
                "best_by_task": {"obstacle_avoidance": candidate_record()},
                "best_multitask": candidate_record(label="multi", tasks=["position_yaw", "obstacle_avoidance"], latency=None),
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
    assert report["global_evidence"]["replay_comparison"]["passed"] is True
    assert report["global_evidence"]["residual_sweep"]["best"]["name"] == "scale005"
    assert report["global_evidence"]["training_throughput"]["best_total_sps"]["total_sps"] == 123456.0
    assert report["global_evidence"]["puffer_export"]["passed"] is True
    assert report["records"][0]["sim"]["mean_yaw_error_rad"] == 0.05
    assert report["records"][0]["sim"]["yaw_error_p95_rad"] == 0.07
    assert report["records"][0]["sim"]["per_task_gate"]["obstacle_avoidance"]["passed"] is True
    assert report["summary"]["ready_tasks"] == ["obstacle_avoidance"]
    assert report["records"][1]["task"] == "multitask"
    assert report["records"][1]["tasks"] == ["position_yaw", "obstacle_avoidance"]
    assert "edge_latency_missing" in report["records"][1]["failures"]
    assert output.with_suffix(".md").exists()


def test_readiness_report_surfaces_missing_evidence() -> None:
    evidence = {
        "room": {"present": False, "mapping_ready": False},
        "native_parity": {"present": False, "passed": False},
        "replay_comparison": {"present": False, "required": True, "passed": False},
    }
    record = READINESS.evaluate_record(("position_yaw", candidate_record(passed=False, parity=False, latency=None)), evidence, 50.0)

    assert record["ready"] is False
    assert {"sim_gate", "edge_parity", "edge_latency_missing", "room_map", "native_parity", "replay_comparison_missing"}.issubset(record["failures"])


def test_readiness_report_blocks_failed_profile_matrix() -> None:
    evidence = {
        "room": {"present": True, "mapping_ready": True},
        "native_parity": {"present": True, "passed": True},
        "replay_comparison": {"present": False, "required": False, "passed": True},
        "profile_matrix": READINESS.compact_profile_matrix(profile_matrix(passed=False)),
    }
    record = READINESS.evaluate_record(("position_yaw", candidate_record(tasks=["position_yaw"])), evidence, 50.0)

    assert record["ready"] is False
    assert "profile_matrix" in record["failures"]
    assert record["profile_matrix"]["worst_survival_fraction"] == 0.7


def test_readiness_report_requires_position_yaw_profile_evidence_when_matrix_present() -> None:
    evidence = {
        "room": {"present": True, "mapping_ready": True},
        "native_parity": {"present": True, "passed": True},
        "replay_comparison": {"present": False, "required": False, "passed": True},
        "profile_matrix": READINESS.compact_profile_matrix({"profiles": ["broad"], "records": []}),
    }
    record = READINESS.evaluate_record(("position_yaw", candidate_record(tasks=["position_yaw"])), evidence, 50.0)

    assert "profile_matrix_missing" in record["failures"]


def test_readiness_report_rejects_bad_replay_comparison() -> None:
    args = argparse_like(require_replay_comparison=False, max_replay_state_rmse=0.5, max_replay_range_rmse_mm=300.0, min_replay_overlap_s=1.0)
    replay = READINESS.compact_replay_comparison(replay_report(state_rmse=0.1, range_rmse=600.0), args)

    assert replay["passed"] is False
    assert "range_rmse" in replay["failures"]


def test_readiness_report_rejects_native_termination_mismatch() -> None:
    compact = READINESS.compact_native_parity(native_report(state_rmse=1e-8, range_rmse=0.1, mismatches=1), 1e-5, 1.0)

    assert compact["passed"] is False
    assert "termination_mismatch" in compact["failures"]


def test_readiness_report_can_require_training_throughput() -> None:
    evidence = {
        "room": {"present": True, "mapping_ready": True},
        "native_parity": {"present": True, "passed": True},
        "replay_comparison": {"present": False, "required": False, "passed": True},
        "training_throughput": READINESS.compact_training_throughput(throughput_report(total_sps=123.0)),
    }

    record = READINESS.evaluate_record(("obstacle_avoidance", candidate_record()), evidence, 50.0, True, 1000.0)

    assert record["ready"] is False
    assert "training_throughput_slow" in record["failures"]


def test_readiness_report_reports_missing_required_training_throughput() -> None:
    evidence = {
        "room": {"present": True, "mapping_ready": True},
        "native_parity": {"present": True, "passed": True},
        "replay_comparison": {"present": False, "required": False, "passed": True},
        "training_throughput": {"present": False},
    }

    record = READINESS.evaluate_record(("obstacle_avoidance", candidate_record()), evidence, 50.0, True, 1.0)

    assert "training_throughput_missing" in record["failures"]


def test_readiness_candidates_include_multitask_after_single_tasks() -> None:
    matrix = {
        "best_by_task": {"position_yaw": candidate_record(label="single")},
        "best_multitask": candidate_record(label="multi", tasks=["position_yaw", "circle"]),
    }

    assert [key for key, _ in READINESS.readiness_candidates(matrix)] == ["position_yaw", "multitask"]


def candidate_record(*, label: str = "candidate", passed: bool = True, parity: bool = True, latency: float | None = 9.0, tasks: list[str] | None = None) -> dict:
    return {
        "label": label,
        "checkpoint": "candidate.pt",
        "tasks": tasks or ["obstacle_avoidance"],
        "passed": passed,
        "failures": [] if passed else ["position_error"],
        "mean_completed_fraction": 1.0,
        "mean_position_error_m": 0.1,
        "mean_yaw_error_rad": 0.05,
        "yaw_error_p95_rad": 0.07,
        "clearance_p01_m": 0.5,
        "edge_parity": {"present": parity, "passed": parity},
        "edge_latency": {"present": latency is not None, **({"per_sample_us": latency} if latency is not None else {})},
        "per_task_gate": {task: {"passed": True, "failures": []} for task in (tasks or ["obstacle_avoidance"])},
    }


def room_report(*, mapping_ready: bool) -> dict:
    return {
        "summary": {"mapping_ready": mapping_ready, "failures": [], "point_count": 100, "duration_s": 10.0},
        "room_estimate": {"width_m": 2.0, "depth_m": 3.0, "height_m": 2.5, "warnings": []},
    }


def native_report(*, state_rmse: float, range_rmse: float, mismatches: int = 0) -> dict:
    return {
        "aligned": {
            "samples": 10,
            "overlap_duration_s": 1.0,
            "signals": {
                "stateEstimate.x": {"rmse": state_rmse},
                "range.front": {"rmse": range_rmse},
            },
        },
        "profiles": [{"terminal_mismatches": mismatches, "truncation_mismatches": 0}],
    }


def replay_report(*, state_rmse: float, range_rmse: float, overlap: float = 2.0) -> dict:
    return {
        "aligned": {
            "samples": 20,
            "overlap_duration_s": overlap,
            "signals": {
                "stateEstimate.x": {"rmse": state_rmse},
                "range.front": {"rmse": range_rmse},
            },
        }
    }


def residual_sweep_report() -> dict:
    return {
        "run": True,
        "thresholds": {"max_teacher_action_l2_mean": 0.02},
        "summary": {
            "total": 1,
            "completed": 1,
            "best": {
                "name": "scale005",
                "checkpoint": "residual.pt",
                "passed": True,
                "mean_completed_fraction": 1.0,
                "mean_position_error_m": 0.19,
                "mean_yaw_error_rad": 0.2,
                "teacher_action_l2_mean": 0.001,
            },
        },
    }


def throughput_report(total_sps: float = 123456.0) -> dict:
    return {
        "controller": "teacher_residual",
        "residual_scale": 0.05,
        "tasks": ["circle"],
        "summary": {"total": 1, "best_total_sps": {"name": "base", "total_sps": total_sps, "num_envs": 256, "horizon": 32, "hidden_size": 128}},
    }


def puffer_export_report(*, passed: bool = True) -> dict:
    return {"passed": passed, "env_name": "flightrl_sixdof", "checks": [{"name": "binding_tokens", "passed": passed}]}


def profile_matrix(*, passed: bool) -> dict:
    return {
        "profiles": ["position_yaw_recovery", "broad"],
        "records": [
            {
                "label": "candidate",
                "checkpoint": "candidate.pt",
                "passed_all_profiles": passed,
                "missing_profiles": [],
                "failures_by_profile": {} if passed else {"broad": ["completion"]},
                "worst_survival_fraction": 0.7,
                "worst_completed_fraction": 0.3,
                "worst_position_error_m": 2.0,
                "worst_clearance_p01_m": 0.05,
                "worst_yaw_error_rad": 0.2,
            }
        ],
    }


def argparse_like(**kwargs):
    return type("Args", (), kwargs)()
