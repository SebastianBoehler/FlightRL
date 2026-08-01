from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

from flightrl.sim2real.measured_config import build_measured_hardware_config
from flightrl.sim2real.hardware_config import summarize_hardware_model


def test_build_measured_hardware_config_maps_sim_alignment_without_claiming_measurement(tmp_path: Path) -> None:
    base = write_base(tmp_path / "base.toml")
    physics = write_json(
        tmp_path / "physics.json",
        {
            "passed": True,
            "failures": [],
            "best": {"physics_profile": {"mass_kg": 0.036, "linear_drag": 0.04, "rate_tau_s": 0.06, "motor_tau_s": 0.02}},
        },
    )
    motor = write_json(tmp_path / "motor.json", {"summary": {"passed": True, "failures": [], "gain_imbalance": 0.12}})
    system_id = write_json(
        tmp_path / "system_id.json",
        {
            "summary": {"profile_ready": True, "failures": [], "tracking_runs": 1, "tracking_samples": 64},
            "response": {"lag_s": {"median": 0.14}, "tau_s": {"median": 0.45}, "gain": {"median": 0.61}},
        },
    )
    sensors = write_json(
        tmp_path / "sensors.json",
        {
            "summary": {"profile_ready": True, "failures": []},
            "sensor_profile": {
                "state_noise_std_m": 0.001,
                "body_rate_noise_std_rad_s": 0.01,
                "range_noise_std_m": 0.002,
                "range_dropout_prob": 0.01,
                "action_lag_s": 0.03,
            },
        },
    )
    output = tmp_path / "measured.toml"

    report = build_measured_hardware_config(
        base_config=base,
        output_config=output,
        physics_calibration=physics,
        motor_calibration=motor,
        live_system_id=system_id,
        sensor_profile=sensors,
    )
    config = tomllib.loads(output.read_text())

    assert config["sim2real"]["measured"] is False
    assert config["sim2real"]["sim_alignment_parameters"] == ["actuator_tau", "drag", "mass"]
    assert config["sim2real"]["unmeasured_dynamics_parameters"] == sorted(config["drone"])
    assert "measured_hardware_dynamics_missing" in config["sim2real"]["measurement_failures"]
    assert config["drone"]["mass"] == 0.036
    assert config["drone"]["drag"] == 0.04
    assert config["drone"]["actuator_tau"] == 0.06
    assert config["sensors"]["range_noise_std_m"] == 0.002
    assert config["domain_randomization"]["thrust_scale"] == 0.12
    assert report["sim2real"]["response_tau_s"] == 0.45
    assert report["parameter_sources"]["inertia"] == "base_config"
    assert report["parameter_sources"]["drag"] == "sim_alignment"


def test_build_measured_hardware_config_rejects_truthy_evidence_flags(tmp_path: Path) -> None:
    base = write_base(tmp_path / "base.toml")
    physics = write_json(
        tmp_path / "physics.json",
        {
            "passed": "true",
            "failures": [],
            "best": {"physics_profile": {"mass_kg": 0.5, "linear_drag": 0.5, "rate_tau_s": 0.5}},
        },
    )
    output = tmp_path / "composed.toml"

    report = build_measured_hardware_config(base_config=base, output_config=output, physics_calibration=physics)
    config = tomllib.loads(output.read_text())

    assert config["sim2real"]["measured"] is False
    assert config["drone"]["mass"] == 0.035
    assert report["parameter_sources"]["mass"] == "base_config"
    assert "physics_sim_alignment_not_ready" in report["sim2real"]["measurement_failures"]


def test_build_measured_hardware_config_cli(tmp_path: Path) -> None:
    base = write_base(tmp_path / "base.toml")
    report = tmp_path / "report.json"
    output = tmp_path / "measured.toml"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_measured_hardware_config.py",
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

    config = tomllib.loads(output.read_text())
    assert "confidence=non_authoritative_composed_profile" in result.stdout
    assert config["sim2real"]["measured"] is False
    assert config["sim2real"]["evidence_files"] == []
    assert summarize_hardware_model(output)["measured"] is False
    assert output.exists()
    assert report.exists()
    assert report.with_suffix(".md").exists()


def write_base(path: Path) -> Path:
    path.write_text(
        """
[environment]
dt = 0.02

[drone]
mass = 0.035
inertia = 0.00003
arm_length = 0.046
drag = 0.05
angular_drag = 0.03
hover_thrust = 0.34335
thrust_gain = 0.16
max_total_thrust = 0.75
max_pitch_torque = 0.015
actuator_tau = 0.06

[sensors]
include_noisy_state = false

[domain_randomization]
enabled = true
thrust_scale = 0.08
""".strip()
        + "\n"
    )
    return path


def write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path
