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
    output = tmp_path / "readiness.json"
    matrix.write_text(json.dumps({"best_by_task": {"obstacle_avoidance": candidate_record()}}))
    room.write_text(json.dumps(room_report(mapping_ready=True)))
    native.write_text(json.dumps(native_report(state_rmse=1e-8, range_rmse=0.1)))

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
    assert report["summary"]["ready_tasks"] == ["obstacle_avoidance"]
    assert output.with_suffix(".md").exists()


def test_readiness_report_surfaces_missing_evidence() -> None:
    evidence = {
        "room": {"present": False, "mapping_ready": False},
        "native_parity": {"present": False, "passed": False},
    }
    record = READINESS.evaluate_record(("position_yaw", candidate_record(passed=False, parity=False, latency=None)), evidence, 50.0)

    assert record["ready"] is False
    assert {"sim_gate", "edge_parity", "edge_latency_missing", "room_map", "native_parity"}.issubset(record["failures"])


def candidate_record(*, passed: bool = True, parity: bool = True, latency: float | None = 9.0) -> dict:
    return {
        "label": "candidate",
        "checkpoint": "candidate.pt",
        "passed": passed,
        "failures": [] if passed else ["position_error"],
        "mean_completed_fraction": 1.0,
        "mean_position_error_m": 0.1,
        "clearance_p01_m": 0.5,
        "edge_parity": {"present": parity, "passed": parity},
        "edge_latency": {"present": latency is not None, **({"per_sample_us": latency} if latency is not None else {})},
    }


def room_report(*, mapping_ready: bool) -> dict:
    return {
        "summary": {"mapping_ready": mapping_ready, "failures": [], "point_count": 100, "duration_s": 10.0},
        "room_estimate": {"width_m": 2.0, "depth_m": 3.0, "height_m": 2.5, "warnings": []},
    }


def native_report(*, state_rmse: float, range_rmse: float) -> dict:
    return {
        "aligned": {
            "samples": 10,
            "overlap_duration_s": 1.0,
            "signals": {
                "stateEstimate.x": {"rmse": state_rmse},
                "range.front": {"rmse": range_rmse},
            },
        }
    }
