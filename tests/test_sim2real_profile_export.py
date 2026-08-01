from __future__ import annotations

import json
import subprocess
import sys

import pytest

from flightrl.config import load_config
from flightrl.sim2real.profile_export import export_config


def test_export_refuses_blocked_profile(tmp_path) -> None:
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": False, "failures": ["motor_calibration_failed"]}})
    base = write_base_config(tmp_path)
    output = tmp_path / "measured.toml"

    report = export_config(profile, base_config=base, output_config=output)

    assert report["exported"] is False
    assert "profile_not_ready" in report["failures"]
    assert not output.exists()


@pytest.mark.parametrize("profile_ready", ["false", "true", 1])
def test_export_rejects_non_boolean_profile_ready(tmp_path, profile_ready) -> None:
    profile_data = ready_profile()
    profile_data["summary"]["profile_ready"] = profile_ready
    profile = write_json(tmp_path / "profile.json", profile_data)
    output = tmp_path / "measured.toml"

    report = export_config(profile, base_config=write_base_config(tmp_path), output_config=output)

    assert report["exported"] is False
    assert report["failures"] == ["profile_not_ready"]
    assert not output.exists()


@pytest.mark.parametrize(
    ("failures", "expected"),
    [
        (["motor_calibration_failed"], ["motor_calibration_failed"]),
        ("none", ["profile_failures_invalid"]),
    ],
)
def test_export_rejects_ready_profile_with_invalid_or_nonempty_failures(tmp_path, failures, expected) -> None:
    profile_data = ready_profile()
    profile_data["summary"]["failures"] = failures
    profile = write_json(tmp_path / "profile.json", profile_data)
    output = tmp_path / "measured.toml"

    report = export_config(profile, base_config=write_base_config(tmp_path), output_config=output)

    assert report["exported"] is False
    assert report["failures"] == ["profile_not_ready", *expected]
    assert not output.exists()


@pytest.mark.parametrize("mass", [float("nan"), float("inf"), "tiny", True, -1.0])
def test_export_rejects_invalid_overlay_dynamics(tmp_path, mass) -> None:
    profile_data = ready_profile()
    profile_data["simulator_overlay"]["drone"]["mass"] = mass
    profile = write_json(tmp_path / "profile.json", profile_data)
    output = tmp_path / "measured.toml"

    report = export_config(profile, base_config=write_base_config(tmp_path), output_config=output)

    assert report["exported"] is False
    assert report["failures"] == ["simulator_overlay_drone_invalid"]
    assert not output.exists()


@pytest.mark.parametrize(
    ("section", "value", "failure"),
    [
        ("sensors", [], "simulator_overlay_sensors_invalid"),
        ("sensors", {"state_noise_std": float("nan")}, "simulator_overlay_sensors_invalid"),
        ("sensors", {"junk": 1.0}, "simulator_overlay_sensors_invalid"),
        ("domain_randomization", "enabled", "simulator_overlay_domain_randomization_invalid"),
        (
            "domain_randomization",
            {"enabled": True, "sensor_noise_scale": "wide"},
            "simulator_overlay_domain_randomization_invalid",
        ),
        ("actuator", None, "simulator_overlay_actuator_invalid"),
        (
            "actuator",
            {"present": True, "relative_motor_gains": {"1": "one"}},
            "simulator_overlay_actuator_invalid",
        ),
    ],
)
def test_export_rejects_malformed_overlay_sections(tmp_path, section, value, failure) -> None:
    profile_data = ready_profile()
    profile_data["simulator_overlay"][section] = value
    profile = write_json(tmp_path / "profile.json", profile_data)

    report = export_config(
        profile,
        base_config=write_base_config(tmp_path),
        output_config=tmp_path / "measured.toml",
    )

    assert report["exported"] is False
    assert failure in report["failures"]


def test_export_writes_loadable_toml_for_ready_profile(tmp_path) -> None:
    profile = write_json(tmp_path / "profile.json", ready_profile())
    base = write_base_config(tmp_path)
    output = tmp_path / "measured.toml"

    report = export_config(profile, base_config=base, output_config=output)
    config = load_config(output)

    assert report["exported"] is True
    assert config.drone.mass == 0.035
    assert config.sensors.include_noisy_state is True
    assert config.sensors.state_noise_std == 0.004
    assert config.domain_randomization.sensor_noise_scale == 1.0


def test_export_cli_writes_report(tmp_path) -> None:
    profile = write_json(tmp_path / "profile.json", {"summary": {"profile_ready": False, "failures": ["blocked"]}})
    base = write_base_config(tmp_path)
    output = tmp_path / "measured.toml"
    report = tmp_path / "export.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_sim2real_config.py",
            "--profile",
            str(profile),
            "--base-config",
            str(base),
            "--output-config",
            str(output),
            "--report",
            str(report),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "exported=False" in result.stdout
    assert report.exists()
    assert report.with_suffix(".md").exists()


def ready_profile():
    return {
        "summary": {"profile_ready": True, "failures": []},
        "simulator_overlay": {
            "drone": {
                "mass": 0.035,
                "inertia": 0.00003,
                "arm_length": 0.046,
                "drag": 0.05,
                "angular_drag": 0.03,
                "hover_thrust": 0.34335,
                "thrust_gain": 0.16,
                "max_total_thrust": 0.75,
                "max_pitch_torque": 0.015,
                "actuator_tau": 0.06,
            },
            "actuator": {
                "present": True,
                "relative_motor_gains": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0},
            },
            "sensors": {
                "state_noise_std": 0.004,
                "attitude_noise_std_deg": 0.1,
                "imu_noise_std": 0.02,
                "command_latency_s": 0.08,
                "range_noise_std_mm": 12.0,
            },
            "domain_randomization": {"enabled": True, "sensor_noise_scale": 1.0},
        },
    }


def write_base_config(tmp_path):
    path = tmp_path / "base.toml"
    path.write_text(
        """
[environment]
action_mode = "motor_quad"

[drone]
mass = 1.0
inertia = 0.08
arm_length = 0.25
drag = 0.12
angular_drag = 0.08
hover_thrust = 9.81
thrust_gain = 4.5
max_total_thrust = 18.0
max_pitch_torque = 2.5
actuator_tau = 0.08

[sensors]
include_position = true
include_velocity = true
state_noise_std = 0.01
imu_noise_std = 0.02

[domain_randomization]
enabled = false
sensor_noise_scale = 0.5
""".strip()
    )
    return path


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path
