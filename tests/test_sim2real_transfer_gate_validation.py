from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from flightrl.sim2real.transfer_gate import build_transfer_gate
from sim2real_transfer_gate_test_support import valid_inputs, write_json


def test_transfer_gate_rejects_declared_readiness_failure_and_invalid_task(tmp_path: Path) -> None:
    inputs = valid_inputs(tmp_path)
    readiness = json.loads(inputs["deployment_readiness"].read_text())
    readiness["summary"]["failures"] = ["stale_evidence"]
    readiness["records"][0]["task"] = ""
    inputs["deployment_readiness"] = write_json(inputs["deployment_readiness"], readiness)

    report = build_transfer_gate(**inputs)

    assert report["transfer_approved"] is False
    assert "stale_evidence" in report["summary"]["failures"]
    assert "record_0:invalid_task" in report["summary"]["failures"]


def test_transfer_gate_rejects_unscoped_desktop_and_malformed_blocker_evidence(tmp_path: Path) -> None:
    inputs = valid_inputs(tmp_path)
    inputs["sim_readiness"] = write_json(
        tmp_path / "sim.json",
        {"summary": {"total": 1, "ready": 1, "blocked": 0}, "records": [{"task": "edge-v3", "ready": True, "failures": []}]},
    )
    inputs["hardware_blockers"] = write_json(tmp_path / "blockers.json", {"blockers": [{"unsafe": True}]})

    report = build_transfer_gate(**inputs)

    assert report["transfer_approved"] is False
    assert "desktop_scope_invalid" in report["summary"]["failures"]
    assert "hardware_blockers_invalid" in report["summary"]["failures"]


def test_transfer_gate_rejects_generic_deployment_evidence(tmp_path: Path) -> None:
    inputs = valid_inputs(tmp_path)
    readiness = json.loads(inputs["deployment_readiness"].read_text())
    readiness["evidence_scope"] = "desktop_development"
    readiness["deployment_authority"] = False
    inputs["deployment_readiness"] = write_json(
        inputs["deployment_readiness"], readiness
    )

    report = build_transfer_gate(**inputs)

    assert report["transfer_approved"] is False
    assert "deployment_scope_invalid" in report["summary"]["failures"]


def test_transfer_gate_rejects_stale_deployment_artifact_identity(tmp_path: Path) -> None:
    inputs = valid_inputs(tmp_path)
    readiness = json.loads(inputs["deployment_readiness"].read_text())
    readiness["records"][0]["bundle_identity"]["sha256"] = "0" * 64
    inputs["deployment_readiness"] = write_json(
        inputs["deployment_readiness"], readiness
    )

    report = build_transfer_gate(**inputs)

    assert report["transfer_approved"] is False
    assert "record_0:bundle_identity_invalid" in report["summary"]["failures"]


def test_transfer_gate_checks_room_safety_and_hardware_blockers(
    tmp_path: Path,
) -> None:
    inputs = valid_inputs(tmp_path)
    inputs["room_report"] = write_json(
        tmp_path / "room.json",
        {
            "summary": {
                "mapping_ready": False,
                "failures": ["speed_glitch"],
                "point_count": 100,
            },
            "room_estimate": {"width_m": 2.0},
        },
    )
    inputs["live_safety"] = write_json(
        tmp_path / "safety.json",
        {"summary": {"passed": False, "failures": ["unsafe_script"]}},
    )
    inputs["hardware_blockers"] = write_json(
        tmp_path / "blockers.json",
        {"blockers": ["range_deck_damaged"]},
    )

    report = build_transfer_gate(**inputs)

    assert report["transfer_approved"] is False
    assert {item["name"] for item in report["checks"][-3:]} == {
        "room_map",
        "live_hardware_safety",
        "hardware_blockers",
    }
    assert "speed_glitch" in report["summary"]["failures"]
    assert "unsafe_script" in report["summary"]["failures"]
    assert "range_deck_damaged" in report["summary"]["failures"]


def test_transfer_gate_cli_writes_report(tmp_path: Path) -> None:
    inputs = valid_inputs(tmp_path)
    inputs["audit"] = write_json(
        tmp_path / "audit.json",
        {"transfer_ready": False, "blocking_items": ["blocked"]},
    )
    output = tmp_path / "gate.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_sim2real_transfer_gate.py",
            "--audit",
            str(inputs["audit"]),
            "--profile",
            str(inputs["profile"]),
            "--config-export",
            str(inputs["config_export"]),
            "--deployment-readiness",
            str(inputs["deployment_readiness"]),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "transfer_approved=False" in result.stdout
    assert output.exists()
    assert output.with_suffix(".md").exists()
