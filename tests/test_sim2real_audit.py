from __future__ import annotations

import csv
import json
import subprocess
import sys

from flightrl.evidence_scope import EDGE_DEPLOYMENT_VERIFIER_MISSING
from flightrl.sim2real.audit import build_audit, render_markdown
from sim2real_audit_test_support import (
    replay_signals,
    stationary_signals,
    write_config,
    write_live_sensor_profile,
)


def test_audit_blocks_placeholder_and_missing_evidence(tmp_path) -> None:
    config = write_config(tmp_path, "manufacturer_placeholder.toml", measured=False)
    report = build_audit(hardware_config=config)

    assert report["transfer_ready"] is False
    assert "measured_dynamics_missing" in report["blocking_items"]
    assert "motor_bench_missing" in report["blocking_items"]
    assert "replay_comparison_missing" in report["blocking_items"]
    assert report["hardware_config"]["parameters"]["mass"] == 1.15


def test_audit_requires_explicit_measured_metadata(tmp_path) -> None:
    config = write_config(tmp_path, "crazyflie_measured.toml", measured=False)

    report = build_audit(hardware_config=config)

    assert report["hardware_config"]["source"] == "test_fixture"
    assert report["hardware_config"]["measured"] is False
    assert "measured_dynamics_missing" in report["blocking_items"]


def test_audit_accepts_complete_motor_bench(tmp_path) -> None:
    config = write_config(tmp_path, "crazyflie_measured.toml", measured=True)
    motor_bench = tmp_path / "motor_bench.csv"
    with motor_bench.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["motor", "power", "rpm", "motor_output", "motor_requested", "vbat"])
        writer.writeheader()
        for motor in range(1, 5):
            for power in [14000, 20000, 26000]:
                writer.writerow({"motor": motor, "power": power, "rpm": power + motor, "motor_output": power, "motor_requested": power, "vbat": 4.0})

    report = build_audit(hardware_config=config, motor_bench=motor_bench)

    assert report["motor_bench"]["passed"] is True
    assert "motor_bench_missing" not in report["blocking_items"]
    assert "motor_bench_failed" not in report["blocking_items"]


def test_audit_reports_failed_replay_and_calibration(tmp_path) -> None:
    config = write_config(tmp_path, "crazyflie_measured.toml", measured=True)
    calibration = tmp_path / "quality.json"
    replay = tmp_path / "replay.json"
    calibration.write_text(json.dumps({"summary": {"replay_calibration_ready": False, "failures": ["floor_range"]}}))
    replay.write_text(
        json.dumps(
            {
                "aligned": {
                    "samples": 10,
                    "overlap_duration_s": 1.0,
                    "signals": replay_signals(10, 0.6, 1000.0),
                }
            }
        )
    )

    report = build_audit(hardware_config=config, calibration_quality=calibration, replay_comparison=replay)

    assert "calibration_flight_not_ready" in report["blocking_items"]
    assert "replay_comparison_failed" in report["blocking_items"]
    assert report["replay_comparison"]["worst_state_rmse"] == 0.6


def test_audit_consumes_sensor_noise_and_latency_evidence(tmp_path) -> None:
    config = write_config(tmp_path, "crazyflie_measured.toml", measured=True)
    noise = tmp_path / "noise.json"
    latency = tmp_path / "latency.json"
    noise.write_text(
        json.dumps(
            {
                "summary": {
                    "stationary_noise_ready": True,
                    "failures": [],
                    "rows": 100,
                    "duration_s": 60.0,
                    "sample_rate_hz": 10.0,
                    "max_position_span_m": 0.01,
                    "max_attitude_span_deg": 0.2,
                },
                "signals": stationary_signals(),
            }
        )
    )
    latency.write_text(json.dumps({"summary": {"latency_ready": True, "failures": [], "accepted_pairs": 2, "median_latency_s": 0.08}}))

    report = build_audit(hardware_config=config, stationary_noise=noise, hardware_latency=latency)

    assert "sensor_noise_unmeasured" not in report["blocking_items"]
    assert "latency_unmeasured" not in report["blocking_items"]
    assert report["stationary_noise"]["passed"] is True
    assert report["hardware_latency"]["passed"] is True


def test_audit_accepts_external_sensor_profile(tmp_path) -> None:
    config = write_config(tmp_path, "crazyflie_measured.toml", measured=True, include_noisy_state=False)
    sensor_profile = write_live_sensor_profile(tmp_path)

    report = build_audit(hardware_config=config, sensor_profile=sensor_profile)

    assert report["sensor_profile"]["passed"] is True
    assert report["sensor_profile"]["action_lag_s"] > 0.0
    assert "sensor_model_incomplete" not in report["blocking_items"]


def test_audit_blocks_empty_deployment_readiness(tmp_path) -> None:
    config = write_config(tmp_path, "crazyflie_measured.toml", measured=True)
    deployment = tmp_path / "deployment_readiness.json"
    deployment.write_text(
        json.dumps(
            {
                "schema": "flightrl.edge_v3.deployment_readiness.v1",
                "target": "ai_deck_gap8",
                "evidence_scope": "edge_deployment",
                "deployment_authority": True,
                "summary": {"total": 0, "ready": 0, "blocked": 0},
                "records": [],
            }
        )
    )

    report = build_audit(hardware_config=config, deployment_readiness=deployment)

    assert report["deployment_readiness"]["passed"] is False
    assert report["deployment_readiness"]["failures"] == [
        EDGE_DEPLOYMENT_VERIFIER_MISSING,
        "no_candidates",
    ]
    assert "deployment_readiness_blocked" in report["blocking_items"]


def test_render_markdown_contains_blockers(tmp_path) -> None:
    report = build_audit(hardware_config=write_config(tmp_path, "manufacturer_placeholder.toml", measured=False))
    markdown = render_markdown(report)

    assert "Transfer ready: `False`" in markdown
    assert "`measured_dynamics_missing`" in markdown


def test_cli_writes_json_and_markdown(tmp_path) -> None:
    config = write_config(tmp_path, "manufacturer_placeholder.toml", measured=False)
    output = tmp_path / "audit.json"

    result = subprocess.run(
        [sys.executable, "scripts/build_sim2real_audit.py", "--hardware-config", str(config), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "transfer_ready=False" in result.stdout
    assert output.exists()
    assert output.with_suffix(".md").exists()
