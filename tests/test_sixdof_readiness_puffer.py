from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_sixdof_readiness_report", ROOT / "scripts" / "build_sixdof_readiness_report.py")
assert SPEC and SPEC.loader
READINESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = READINESS
SPEC.loader.exec_module(READINESS)


def test_readiness_report_can_require_puffer_export() -> None:
    evidence = evidence_with_puffer(passed=False)

    record = READINESS.evaluate_record(("obstacle_avoidance", candidate_record()), evidence, 50.0, require_puffer_export=True)

    assert record["ready"] is False
    assert "puffer_export" in record["failures"]


def test_readiness_report_reports_missing_required_puffer_export() -> None:
    evidence = base_evidence()
    evidence["puffer_export"] = {"present": False}

    record = READINESS.evaluate_record(("obstacle_avoidance", candidate_record()), evidence, 50.0, require_puffer_export=True)

    assert "puffer_export_missing" in record["failures"]


def evidence_with_puffer(*, passed: bool) -> dict:
    evidence = base_evidence()
    evidence["puffer_export"] = READINESS.compact_puffer_export({"passed": passed, "env_name": "flightrl_sixdof"})
    return evidence


def base_evidence() -> dict:
    return {
        "room": {"present": True, "mapping_ready": True},
        "native_parity": {"present": True, "passed": True},
        "replay_comparison": {"present": False, "required": False, "passed": True},
    }


def candidate_record() -> dict:
    return {
        "label": "candidate",
        "checkpoint": "candidate.pt",
        "tasks": ["obstacle_avoidance"],
        "passed": True,
        "mean_completed_fraction": 1.0,
        "mean_position_error_m": 0.1,
        "clearance_p01_m": 0.5,
        "edge_parity": {"present": True, "passed": True},
        "edge_latency": {"present": True, "per_sample_us": 9.0},
    }
