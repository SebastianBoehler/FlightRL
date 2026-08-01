from __future__ import annotations

import json

import pytest

from flightrl.sim2real.audit import build_audit
from flightrl.sim2real.noise import DEFAULT_COLUMNS
from sim2real_audit_test_support import write_config


def test_audit_rejects_truthy_string_measured_metadata(tmp_path) -> None:
    config = write_config(tmp_path, "crazyflie_measured.toml", measured=True)
    config.write_text(config.read_text().replace("measured = true", 'measured = "true"'))

    report = build_audit(hardware_config=config)

    assert report["hardware_config"]["measured"] is False
    assert "measured_dynamics_missing" in report["blocking_items"]


def test_audit_rejects_sparse_stationary_noise_claim(tmp_path) -> None:
    config = write_config(tmp_path, "crazyflie_measured.toml", measured=True)
    noise = tmp_path / "sparse_noise.json"
    noise.write_text(
        json.dumps(
            {
                "summary": {
                    "stationary_noise_ready": True,
                    "failures": [],
                    "rows": 2,
                    "duration_s": 30.0,
                    "sample_rate_hz": 1.0 / 30.0,
                    "max_position_span_m": 0.0,
                    "max_attitude_span_deg": 0.0,
                },
                "signals": {
                    column: {"samples": 1, "valid_ratio": 1.0, "std": 0.0}
                    for column in DEFAULT_COLUMNS
                },
            }
        )
    )

    report = build_audit(hardware_config=config, stationary_noise=noise)

    assert report["stationary_noise"]["passed"] is False
    assert "stationary_noise_invalid_metrics" in report["stationary_noise"]["failures"]


def test_audit_rejects_truthy_string_evidence_flags(tmp_path) -> None:
    config = write_config(tmp_path, "crazyflie_measured.toml", measured=True)
    calibration = tmp_path / "quality.json"
    noise = tmp_path / "noise.json"
    latency = tmp_path / "latency.json"
    calibration.write_text(json.dumps({"summary": {"replay_calibration_ready": "false", "failures": []}}))
    noise.write_text(json.dumps({"summary": {"stationary_noise_ready": "false", "failures": []}}))
    latency.write_text(json.dumps({"summary": {"latency_ready": "false", "failures": []}}))

    report = build_audit(
        hardware_config=config,
        calibration_quality=calibration,
        stationary_noise=noise,
        hardware_latency=latency,
    )

    assert report["calibration_quality"]["ready"] is False
    assert report["stationary_noise"]["passed"] is False
    assert report["hardware_latency"]["passed"] is False


@pytest.mark.parametrize(
    "thresholds",
    (
        {"max_replay_state_rmse": float("nan")},
        {"max_replay_range_rmse_mm": float("inf")},
        {"min_motor_powers": True},
    ),
)
def test_audit_rejects_invalid_thresholds(tmp_path, thresholds) -> None:
    config = write_config(tmp_path, "crazyflie_measured.toml", measured=True)

    with pytest.raises(ValueError):
        build_audit(hardware_config=config, **thresholds)


def test_audit_rejects_nonfinite_and_contradictory_evidence(tmp_path) -> None:
    config = write_config(tmp_path, "crazyflie_measured.toml", measured=True)
    replay = tmp_path / "replay.json"
    noise = tmp_path / "noise.json"
    latency = tmp_path / "latency.json"
    replay.write_text(json.dumps({"aligned": {"samples": 10, "overlap_duration_s": 2.0, "signals": {"stateEstimate.x": {"rmse": float("nan")}, "range.front": {"rmse": 10.0}}}}))
    noise.write_text(json.dumps({"summary": {"stationary_noise_ready": True, "failures": ["stale"], "duration_s": 60.0, "sample_rate_hz": 10.0, "max_position_span_m": 0.01, "max_attitude_span_deg": 0.2}}))
    latency.write_text(json.dumps({"summary": {"latency_ready": True, "failures": [], "accepted_pairs": 1, "median_latency_s": float("inf")}}))

    report = build_audit(hardware_config=config, replay_comparison=replay, stationary_noise=noise, hardware_latency=latency)

    assert report["replay_comparison"]["passed"] is False
    assert "state_rmse" in report["replay_comparison"]["failures"]
    assert report["stationary_noise"]["passed"] is False
    assert report["hardware_latency"]["passed"] is False


def test_audit_rejects_invalid_sensor_profile_values(tmp_path) -> None:
    config = write_config(tmp_path, "crazyflie_measured.toml", measured=True, include_noisy_state=False)
    sensor_profile = tmp_path / "sensor_profile.json"
    sensor_profile.write_text(json.dumps({"sensor_profile": {"enabled": True, "range_noise_std_m": "0.01", "range_dropout_prob": 1.5}}))

    report = build_audit(hardware_config=config, sensor_profile=sensor_profile)

    assert report["sensor_profile"]["passed"] is False
    assert {"invalid_values", "invalid_dropout_probability"}.issubset(report["sensor_profile"]["failures"])
    assert "sensor_model_incomplete" in report["blocking_items"]

    sensor_profile.write_text(json.dumps({"sensor_profile": {"enabled": True}}))
    empty = build_audit(hardware_config=config, sensor_profile=sensor_profile)
    assert "empty_profile" in empty["sensor_profile"]["failures"]


def test_audit_rejects_sensor_profile_without_source_provenance(tmp_path) -> None:
    config = write_config(tmp_path, "crazyflie_measured.toml", measured=True, include_noisy_state=False)
    sensor_profile = tmp_path / "fabricated.json"
    sensor_profile.write_text(
        json.dumps(
            {
                "summary": {"profile_ready": True, "failures": [], "flight_rows": 0},
                "inputs": {"flight_logs": [], "stationary_logs": [], "latency_report": None},
                "sensor_profile": {
                    "name": "fabricated",
                    "enabled": True,
                    "state_noise_std_m": 0.01,
                    "velocity_noise_std_m_s": 0.01,
                    "body_rate_noise_std_rad_s": 0.01,
                    "range_noise_std_m": 0.01,
                    "range_dropout_prob": 0.01,
                    "action_lag_s": 0.01,
                },
            }
        )
    )

    report = build_audit(hardware_config=config, sensor_profile=sensor_profile)

    assert report["sensor_profile"]["passed"] is False
    assert "profile_provenance_invalid" in report["sensor_profile"]["failures"]
    assert "sensor_model_incomplete" in report["blocking_items"]


def test_audit_rejects_self_declared_deployment_without_typed_identities(tmp_path) -> None:
    config = write_config(tmp_path, "crazyflie_measured.toml", measured=True)
    deployment = tmp_path / "deployment.json"
    deployment.write_text(
        json.dumps(
            {
                "evidence_scope": "edge_deployment",
                "deployment_authority": True,
                "summary": {"total": 1, "ready": 1, "blocked": 0},
                "records": [
                    {
                        "task": "obstacle_avoidance",
                        "ready": True,
                        "failures": [],
                    }
                ],
            }
        )
    )

    report = build_audit(
        hardware_config=config,
        deployment_readiness=deployment,
    )

    assert report["deployment_readiness"]["passed"] is False
    assert "deployment_schema_invalid" in report["deployment_readiness"]["failures"]
