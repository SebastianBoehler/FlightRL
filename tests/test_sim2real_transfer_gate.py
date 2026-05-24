from __future__ import annotations

import json
import subprocess
import sys

from flightrl.sim2real.transfer_gate import build_transfer_gate


def test_transfer_gate_blocks_failed_evidence(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": False, "blocking_items": ["motor_bench_failed"]})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": False, "failures": ["motor_calibration_failed"]}})
    export = write_json(tmp_path / "export.json", {"exported": False, "failures": ["profile_not_ready"]})
    readiness = write_json(tmp_path / "readiness.json", {"summary": {"total": 1, "ready": 0, "blocked": 1}, "records": [{"task": "multitask", "ready": False, "failures": ["replay_comparison"]}]})

    report = build_transfer_gate(audit=audit, profile=profile, config_export=export, deployment_readiness=readiness)

    assert report["transfer_approved"] is False
    assert "motor_bench_failed" in report["summary"]["failures"]
    assert "multitask:replay_comparison" in report["summary"]["failures"]


def test_transfer_gate_approves_complete_evidence(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": True, "blocking_items": []})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": True, "failures": []}})
    export = write_json(tmp_path / "export.json", {"exported": True, "failures": []})
    readiness = write_json(tmp_path / "readiness.json", {"summary": {"total": 2, "ready": 2, "blocked": 0}, "records": [{"task": "a", "ready": True}, {"task": "b", "ready": True}]})

    report = build_transfer_gate(audit=audit, profile=profile, config_export=export, deployment_readiness=readiness)

    assert report["transfer_approved"] is True
    assert report["summary"]["passed"] == 4


def test_transfer_gate_includes_room_map_check(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": True, "blocking_items": []})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": True, "failures": []}})
    export = write_json(tmp_path / "export.json", {"exported": True, "failures": []})
    readiness = write_json(tmp_path / "readiness.json", {"summary": {"total": 1, "ready": 1, "blocked": 0}, "records": [{"task": "a", "ready": True}]})
    room = write_json(tmp_path / "room.json", {"summary": {"mapping_ready": False, "failures": ["speed_glitch"], "point_count": 100}, "room_estimate": {"width_m": 2.0}})

    report = build_transfer_gate(audit=audit, profile=profile, config_export=export, deployment_readiness=readiness, room_report=room)

    assert report["transfer_approved"] is False
    assert report["checks"][-1]["name"] == "room_map"
    assert "speed_glitch" in report["summary"]["failures"]


def test_transfer_gate_includes_live_hardware_safety_check(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": True, "blocking_items": []})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": True, "failures": []}})
    export = write_json(tmp_path / "export.json", {"exported": True, "failures": []})
    readiness = write_json(tmp_path / "readiness.json", {"summary": {"total": 1, "ready": 1, "blocked": 0}, "records": [{"task": "a", "ready": True}]})
    safety = write_json(tmp_path / "live_safety.json", {"summary": {"passed": False, "failures": ["unsafe.py:checkpoint_control_without_hardware_approval"]}})

    report = build_transfer_gate(audit=audit, profile=profile, config_export=export, deployment_readiness=readiness, live_safety=safety)

    assert report["transfer_approved"] is False
    assert report["checks"][-1]["name"] == "live_hardware_safety"
    assert "unsafe.py:checkpoint_control_without_hardware_approval" in report["summary"]["failures"]


def test_transfer_gate_cli_writes_report(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": False, "blocking_items": ["blocked"]})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": False, "failures": ["blocked"]}})
    export = write_json(tmp_path / "export.json", {"exported": False, "failures": ["blocked"]})
    readiness = write_json(tmp_path / "readiness.json", {"summary": {"total": 1, "ready": 0, "blocked": 1}, "records": []})
    output = tmp_path / "gate.json"
    safety = write_json(tmp_path / "live_safety.json", {"summary": {"passed": True, "failures": []}})

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_sim2real_transfer_gate.py",
            "--audit",
            str(audit),
            "--profile",
            str(profile),
            "--config-export",
            str(export),
            "--deployment-readiness",
            str(readiness),
            "--live-safety",
            str(safety),
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


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path
