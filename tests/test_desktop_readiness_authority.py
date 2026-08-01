from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from flightrl.sim2real.audit import summarize_deployment
from flightrl.sim2real.checkpoint_manifest import build_checkpoint_manifest
from flightrl.sim2real.transfer_gate import build_transfer_gate


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("desktop_readiness", ROOT / "scripts" / "build_sixdof_readiness_report.py")
assert SPEC and SPEC.loader
DESKTOP_READINESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DESKTOP_READINESS
SPEC.loader.exec_module(DESKTOP_READINESS)


def test_desktop_readiness_rejects_legacy_edge_evidence() -> None:
    record = readiness()["records"][0]
    record["edge_parity"] = {"present": True, "passed": True}
    evidence = {
        "room": {"mapping_ready": True},
        "native_parity": {"passed": True},
        "replay_comparison": {"present": False, "required": False},
    }

    with pytest.raises(ValueError, match="legacy edge_parity"):
        DESKTOP_READINESS.evaluate_record(("obstacle_avoidance", record), evidence, 50.0)


def test_desktop_readiness_cannot_pass_deployment_gate(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": True, "blocking_items": []})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": True}})
    config = write_json(tmp_path / "config.json", {"exported": True})
    desktop = write_json(tmp_path / "desktop.json", readiness(scope="desktop_development"))

    report = build_transfer_gate(audit=audit, profile=profile, config_export=config, deployment_readiness=desktop)

    assert report["transfer_approved"] is False
    assert "deployment_scope_invalid" in report["summary"]["failures"]


def test_desktop_readiness_is_rejected_by_sim2real_audit(tmp_path) -> None:
    desktop = write_json(tmp_path / "desktop.json", readiness(scope="desktop_development"))

    deployment = summarize_deployment(desktop)

    assert deployment["passed"] is False
    assert deployment["failures"] == ["deployment_scope_invalid"]


def test_desktop_readiness_cannot_mint_hardware_approved_manifest(tmp_path) -> None:
    gate = write_json(tmp_path / "gate.json", {"transfer_approved": True})
    sim = write_json(tmp_path / "sim.json", readiness())
    desktop = write_json(tmp_path / "desktop.json", readiness(scope="desktop_development"))

    report = build_checkpoint_manifest(transfer_gate=gate, sim_readiness=sim, deployment_readiness=desktop)

    assert report["deployment_authority"] is False
    assert report["summary"]["hardware_approved"] == 0
    assert "deployment_scope_invalid" in report["records"][0]["deployment_failures"]
    assert "checkpoint_missing" in report["records"][0]["deployment_failures"]


def readiness(*, scope: str | None = None) -> dict:
    authority = {"evidence_scope": scope, "deployment_authority": False} if scope else {}
    return {
        **authority,
        "summary": {"total": 1, "ready": 1, "blocked": 0},
        "records": [{"task": "obstacle_avoidance", "checkpoint": "candidate.pt", "ready": True, "failures": []}],
    }


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path
