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


def test_transfer_gate_includes_puffer_transfer_check(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": True, "blocking_items": []})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": True, "failures": []}})
    export = write_json(tmp_path / "export.json", {"exported": True, "failures": []})
    readiness = write_json(tmp_path / "readiness.json", {"summary": {"total": 1, "ready": 1, "blocked": 0}, "records": [{"task": "a", "ready": True}]})
    puffer = write_json(tmp_path / "puffer.json", {"passed": True, "candidates": {"bc": {"passed": True}}})

    report = build_transfer_gate(
        audit=audit,
        profile=profile,
        config_export=export,
        deployment_readiness=readiness,
        puffer_transfer_test=puffer,
    )

    assert report["transfer_approved"] is True
    assert report["checks"][-1]["name"] == "puffer_transfer_test"
    assert report["checks"][-1]["candidates"] == {"bc": True}


def test_transfer_gate_blocks_failed_puffer_transfer_check(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": True, "blocking_items": []})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": True, "failures": []}})
    export = write_json(tmp_path / "export.json", {"exported": True, "failures": []})
    readiness = write_json(tmp_path / "readiness.json", {"summary": {"total": 1, "ready": 1, "blocked": 0}, "records": [{"task": "a", "ready": True}]})
    puffer = write_json(tmp_path / "puffer.json", {"passed": False, "candidates": {"weak": {"passed": False}}})

    report = build_transfer_gate(
        audit=audit,
        profile=profile,
        config_export=export,
        deployment_readiness=readiness,
        puffer_transfer_test=puffer,
    )

    assert report["transfer_approved"] is False
    assert "weak:transfer_test_failed" in report["summary"]["failures"]


def test_transfer_gate_reports_puffer_source_quality_failures(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": True, "blocking_items": []})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": True, "failures": []}})
    export = write_json(tmp_path / "export.json", {"exported": True, "failures": []})
    readiness = write_json(tmp_path / "readiness.json", {"summary": {"total": 1, "ready": 1, "blocked": 0}, "records": [{"task": "a", "ready": True}]})
    puffer = write_json(
        tmp_path / "puffer.json",
        {
            "passed": False,
            "candidates": {"raw90": {"passed": True}},
            "source_teacher_quality": {"vel70": {"gate": {"passed": False, "failures": ["source_teacher_pitch_rate_sign"]}}},
        },
    )

    report = build_transfer_gate(
        audit=audit,
        profile=profile,
        config_export=export,
        deployment_readiness=readiness,
        puffer_transfer_test=puffer,
    )

    assert report["transfer_approved"] is False
    assert "vel70:source_teacher_pitch_rate_sign" in report["summary"]["failures"]


def test_transfer_gate_includes_puffer_bundle_transfer_check(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": True, "blocking_items": []})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": True, "failures": []}})
    export = write_json(tmp_path / "export.json", {"exported": True, "failures": []})
    readiness = write_json(tmp_path / "readiness.json", {"summary": {"total": 1, "ready": 1, "blocked": 0}, "records": [{"task": "a", "ready": True}]})
    puffer = write_json(tmp_path / "bundle.json", {"passed": True, "bundle": {"label": "raw90_vel70", "passed": True}})

    report = build_transfer_gate(
        audit=audit,
        profile=profile,
        config_export=export,
        deployment_readiness=readiness,
        puffer_transfer_test=puffer,
    )

    assert report["transfer_approved"] is True
    assert report["checks"][-1]["candidates"] == {"raw90_vel70": True}


def test_transfer_gate_accepts_multiple_puffer_transfer_checks(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": True, "blocking_items": []})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": True, "failures": []}})
    export = write_json(tmp_path / "export.json", {"exported": True, "failures": []})
    readiness = write_json(tmp_path / "readiness.json", {"summary": {"total": 1, "ready": 1, "blocked": 0}, "records": [{"task": "a", "ready": True}]})
    bundle = write_json(tmp_path / "bundle.json", {"passed": True, "bundle": {"label": "bundle", "passed": True}})
    holdout = write_json(tmp_path / "holdout.json", {"passed": True, "candidates": {"raw90": {"passed": True}}})
    matrix = write_json(tmp_path / "matrix.json", {"passed": True, "label": "matrix", "runs": [{"seed": 1, "passed": True}]})

    report = build_transfer_gate(
        audit=audit,
        profile=profile,
        config_export=export,
        deployment_readiness=readiness,
        puffer_transfer_test=[bundle, holdout, matrix],
    )

    assert report["transfer_approved"] is True
    assert report["checks"][-3]["name"] == "puffer_transfer_test_1"
    assert report["checks"][-2]["name"] == "puffer_transfer_test_2"
    assert report["checks"][-1]["name"] == "puffer_transfer_test_3"
    assert report["checks"][-1]["candidates"] == {"matrix": True}
    assert report["summary"]["passed"] == 7


def test_transfer_gate_blocks_failed_puffer_bundle_transfer_check(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": True, "blocking_items": []})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": True, "failures": []}})
    export = write_json(tmp_path / "export.json", {"exported": True, "failures": []})
    readiness = write_json(tmp_path / "readiness.json", {"summary": {"total": 1, "ready": 1, "blocked": 0}, "records": [{"task": "a", "ready": True}]})
    puffer = write_json(
        tmp_path / "bundle.json",
        {
            "passed": False,
            "bundle": {
                "label": "raw90_vel70",
                "passed": False,
                "obstacle": {"passed": True},
                "velocity": {"vel70": {"gate": {"passed": False}}},
            },
        },
    )

    report = build_transfer_gate(
        audit=audit,
        profile=profile,
        config_export=export,
        deployment_readiness=readiness,
        puffer_transfer_test=puffer,
    )

    assert report["transfer_approved"] is False
    assert "raw90_vel70:velocity_transfer_failed" in report["summary"]["failures"]


def test_transfer_gate_reports_failed_puffer_robustness_matrix(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": True, "blocking_items": []})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": True, "failures": []}})
    export = write_json(tmp_path / "export.json", {"exported": True, "failures": []})
    readiness = write_json(tmp_path / "readiness.json", {"summary": {"total": 1, "ready": 1, "blocked": 0}, "records": [{"task": "a", "ready": True}]})
    matrix = write_json(tmp_path / "matrix.json", {"passed": False, "label": "matrix", "runs": [{"seed": 2, "passed": False, "failures": ["drift"]}]})

    report = build_transfer_gate(audit=audit, profile=profile, config_export=export, deployment_readiness=readiness, puffer_transfer_test=matrix)

    assert report["transfer_approved"] is False
    assert report["checks"][-1]["candidates"] == {"matrix": False}
    assert "matrix:seed_2:drift" in report["summary"]["failures"]


def test_transfer_gate_blocks_hardware_blockers(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": True, "blocking_items": []})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": True, "failures": []}})
    export = write_json(tmp_path / "export.json", {"exported": True, "failures": []})
    readiness = write_json(tmp_path / "readiness.json", {"summary": {"total": 1, "ready": 1, "blocked": 0}, "records": [{"task": "a", "ready": True}]})
    blockers = write_json(tmp_path / "blockers.json", {"blockers": ["range_deck_damaged"]})

    report = build_transfer_gate(
        audit=audit,
        profile=profile,
        config_export=export,
        deployment_readiness=readiness,
        hardware_blockers=blockers,
    )

    assert report["transfer_approved"] is False
    assert report["checks"][-1]["name"] == "hardware_blockers"
    assert "range_deck_damaged" in report["summary"]["failures"]


def test_transfer_gate_deduplicates_summary_failures(tmp_path) -> None:
    audit = write_json(tmp_path / "audit.json", {"transfer_ready": False, "blocking_items": ["range_deck_damaged"]})
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": True, "failures": []}})
    export = write_json(tmp_path / "export.json", {"exported": True, "failures": []})
    readiness = write_json(tmp_path / "readiness.json", {"summary": {"total": 1, "ready": 1, "blocked": 0}, "records": [{"task": "a", "ready": True}]})
    blockers = write_json(tmp_path / "blockers.json", {"blockers": ["range_deck_damaged"]})

    report = build_transfer_gate(
        audit=audit,
        profile=profile,
        config_export=export,
        deployment_readiness=readiness,
        hardware_blockers=blockers,
    )

    assert report["transfer_approved"] is False
    assert report["summary"]["failures"] == ["range_deck_damaged"]
    assert report["checks"][0]["failures"] == ["range_deck_damaged"]
    assert report["checks"][-1]["failures"] == ["range_deck_damaged"]


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
