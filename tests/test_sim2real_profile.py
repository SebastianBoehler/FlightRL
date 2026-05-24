from __future__ import annotations

import json
import subprocess
import sys

from flightrl.sim2real.profile import build_profile


def test_profile_rejects_failed_current_evidence(tmp_path) -> None:
    hardware = write_hardware(tmp_path, "manufacturer_placeholder.toml", measured=False)
    motor = write_json(tmp_path / "motor.json", {"summary": {"passed": False, "failures": ["rpm_signal"]}})
    noise = write_json(tmp_path / "noise.json", {"summary": {"stationary_noise_ready": False, "failures": ["motion"]}})
    latency = write_json(tmp_path / "latency.json", {"summary": {"latency_ready": False, "failures": ["no_accepted_pairs"]}})

    report = build_profile(hardware_config=hardware, motor_calibration=motor, stationary_noise=noise, hardware_latency=latency)

    assert report["summary"]["profile_ready"] is False
    assert "measured_dynamics_missing" in report["summary"]["failures"]
    assert "simulator_overlay" not in report


def test_profile_requires_explicit_measured_metadata(tmp_path) -> None:
    hardware = write_hardware(tmp_path, "measured_crazyflie.toml", measured=False)

    report = build_profile(hardware_config=hardware, motor_calibration=None, stationary_noise=None, hardware_latency=None)

    assert report["hardware_config"]["measured"] is False
    assert "measured_dynamics_missing" in report["summary"]["failures"]


def test_profile_emits_overlay_from_passed_evidence(tmp_path) -> None:
    hardware = write_hardware(tmp_path, "measured_crazyflie.toml", measured=True)
    motor = write_json(
        tmp_path / "motor.json",
        {
            "summary": {"passed": True, "failures": [], "gain_imbalance": 0.04},
            "simulator_priors": {"present": True, "mean_slope_rpm_per_power": 0.45, "relative_motor_gains": {"1": 1.0, "2": 1.01, "3": 0.99, "4": 1.0}},
        },
    )
    noise = write_json(tmp_path / "noise.json", {"summary": {"stationary_noise_ready": True, "failures": []}, "signals": noise_signals()})
    latency = write_json(tmp_path / "latency.json", {"summary": {"latency_ready": True, "failures": [], "median_latency_s": 0.08}})

    report = build_profile(hardware_config=hardware, motor_calibration=motor, stationary_noise=noise, hardware_latency=latency)

    assert report["summary"]["profile_ready"] is True
    assert report["simulator_overlay"]["actuator"]["relative_motor_gains"]["2"] == 1.01
    assert report["simulator_overlay"]["sensors"]["command_latency_s"] == 0.08


def test_profile_cli_writes_report(tmp_path) -> None:
    hardware = write_hardware(tmp_path, "manufacturer_placeholder.toml", measured=False)
    output = tmp_path / "profile.json"

    result = subprocess.run(
        [sys.executable, "scripts/build_sim2real_profile.py", "--hardware-config", str(hardware), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "profile_ready=False" in result.stdout
    assert output.exists()
    assert output.with_suffix(".md").exists()


def write_hardware(tmp_path, name: str, *, measured: bool):
    path = tmp_path / name
    path.write_text(
        f"""
[sim2real]
measured = {str(measured).lower()}
source = "test_fixture"

[drone]
mass = 1.15
inertia = 0.09
arm_length = 0.23
drag = 0.14
angular_drag = 0.09
hover_thrust = 10.6
thrust_gain = 4.2
max_total_thrust = 19.5
max_pitch_torque = 2.2
actuator_tau = 0.11

[domain_randomization]
enabled = true
""".strip()
    )
    return path


def noise_signals():
    signals = {}
    for column, std in {
        "stateEstimate.x": 0.004,
        "stateEstimate.y": 0.005,
        "stateEstimate.z": 0.003,
        "stabilizer.roll": 0.2,
        "stabilizer.pitch": 0.25,
        "stabilizer.yaw": 0.3,
        "acc.x": 0.01,
        "acc.y": 0.02,
        "acc.z": 0.03,
        "gyro.x": 0.1,
        "gyro.y": 0.2,
        "gyro.z": 0.3,
        "range.front": 12.0,
        "range.back": 14.0,
        "range.left": 13.0,
        "range.right": 11.0,
        "range.up": 10.0,
        "range.zrange": 8.0,
    }.items():
        signals[column] = {"samples": 100, "std": std, "span": std * 4}
    return signals


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path
