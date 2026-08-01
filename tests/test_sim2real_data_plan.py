from __future__ import annotations

import csv
import json
import subprocess
import sys

from flightrl.sim2real.data_plan import build_data_plan, render_markdown


def test_data_plan_maps_audit_blockers_to_requirements(tmp_path) -> None:
    audit = write_audit(tmp_path, ["m3_motor_issue", "motor_bench_missing", "sensor_noise_failed"])

    report = build_data_plan(audit)

    by_name = {record["name"]: record for record in report["requirements"]}
    assert by_name["actuator_curve"]["status"] == "hardware_blocked"
    assert by_name["sensor_noise"]["status"] == "blocked"
    assert by_name["latency"]["status"] == "satisfied"


def test_data_plan_maps_failed_latency_to_requirement(tmp_path) -> None:
    audit = write_audit(tmp_path, ["latency_failed"])

    report = build_data_plan(audit)

    latency = {record["name"]: record for record in report["requirements"]}["latency"]
    assert latency["status"] == "blocked"
    assert latency["matched_blockers"] == ["latency_failed"]


def test_data_plan_maps_incomplete_hardware_dynamics_to_measured_dynamics(tmp_path) -> None:
    audit = write_audit(tmp_path, ["hardware_dynamics_incomplete"])

    report = build_data_plan(audit)

    measured = {record["name"]: record for record in report["requirements"]}["measured_dynamics"]
    assert measured["status"] == "blocked"
    assert measured["matched_blockers"] == ["hardware_dynamics_incomplete"]
    assert measured["command"] is None


def test_data_plan_rejects_truthy_ready_and_malformed_blockers(tmp_path) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"transfer_ready": "false", "blocking_items": "none"}))

    report = build_data_plan(audit)

    assert report["transfer_ready"] is False
    assert report["audit_blockers"] == ["audit_blockers_invalid"]


def test_data_plan_keeps_failed_motor_bench_as_partial_evidence(tmp_path) -> None:
    audit = write_audit(tmp_path, ["motor_bench_missing"])
    bench = tmp_path / "motor.csv"
    with bench.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["motor", "power", "rpm", "motor_output", "motor_requested", "vbat"])
        writer.writeheader()
        writer.writerow({"motor": 1, "power": 14000, "rpm": 0, "motor_output": 14000, "motor_requested": 0, "vbat": 3.5})

    report = build_data_plan(audit, motor_bench=bench)

    assert report["partial_evidence"]["motor_bench"]["present"] is True
    assert report["partial_evidence"]["motor_bench"]["passed"] is False
    assert "partial" in report["partial_evidence"]["motor_bench_note"]


def test_data_plan_marks_removed_live_calibration_runner_as_blocked(tmp_path) -> None:
    audit = write_audit(tmp_path, ["calibration_flight_not_ready"])

    markdown = render_markdown(build_data_plan(audit))

    assert "No reviewed collection command is currently available." in markdown
    assert "crazyflie_calibration_flight.py" not in markdown
    assert "Do not run live hardware" in markdown


def test_data_plan_cli_writes_json_and_markdown(tmp_path) -> None:
    audit = write_audit(tmp_path, ["replay_comparison_failed"])
    output = tmp_path / "plan.json"

    result = subprocess.run(
        [sys.executable, "scripts/build_sim2real_data_plan.py", "--audit", str(audit), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "next_actions=" in result.stdout
    assert output.exists()
    assert output.with_suffix(".md").exists()


def write_audit(tmp_path, blockers: list[str]):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"transfer_ready": False, "blocking_items": blockers}))
    return path
