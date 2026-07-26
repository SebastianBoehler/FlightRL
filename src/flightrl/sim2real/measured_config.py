from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from flightrl.sim2real.profile_export import render_toml


PHYSICS_TO_DRONE = {
    "mass_kg": "mass",
    "linear_drag": "drag",
    "rate_tau_s": "actuator_tau",
}


def build_measured_hardware_config(
    *,
    base_config: Path,
    output_config: Path,
    physics_calibration: Path | None = None,
    motor_calibration: Path | None = None,
    live_system_id: Path | None = None,
    sensor_profile: Path | None = None,
) -> dict[str, Any]:
    config = tomllib.loads(base_config.read_text())
    parameter_sources = {key: "base_config" for key in config.get("drone", {})}
    evidence: dict[str, Any] = {"base_config": str(base_config)}

    physics = read_json(physics_calibration)
    if physics:
        evidence["physics_calibration"] = str(physics_calibration)
        apply_physics(config, physics, parameter_sources)

    motor = read_json(motor_calibration)
    if motor:
        evidence["motor_calibration"] = str(motor_calibration)
        apply_motor(config, motor)

    system_id = read_json(live_system_id)
    if system_id:
        evidence["live_system_id"] = str(live_system_id)

    sensors = read_json(sensor_profile)
    if sensors:
        evidence["sensor_profile"] = str(sensor_profile)
        apply_sensor_profile(config, sensors)

    config["sim2real"] = sim2real_section(evidence, parameter_sources, physics, motor, system_id, sensors)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(render_toml(config) + "\n")
    return {
        "output_config": str(output_config),
        "base_config": str(base_config),
        "evidence": evidence,
        "parameter_sources": parameter_sources,
        "sim2real": config["sim2real"],
        "drone": config.get("drone", {}),
        "sensors": config.get("sensors", {}),
    }


def apply_physics(config: dict[str, Any], report: dict[str, Any], parameter_sources: dict[str, str]) -> None:
    profile = report.get("best", {}).get("physics_profile", {})
    drone = config.setdefault("drone", {})
    for source_key, drone_key in PHYSICS_TO_DRONE.items():
        value = profile.get(source_key)
        if value is not None:
            drone[drone_key] = value
            parameter_sources[drone_key] = "physics_calibration"
    if profile.get("thrust_scale") is not None:
        config.setdefault("domain_randomization", {})["thrust_scale"] = max(
            float(config.get("domain_randomization", {}).get("thrust_scale", 0.0) or 0.0),
            0.05,
        )


def apply_motor(config: dict[str, Any], report: dict[str, Any]) -> None:
    summary = report.get("summary", {})
    imbalance = float(summary.get("gain_imbalance", 0.0) or 0.0)
    if imbalance > 0.0:
        config.setdefault("domain_randomization", {})["thrust_scale"] = max(
            float(config.get("domain_randomization", {}).get("thrust_scale", 0.0) or 0.0),
            imbalance,
        )


def apply_sensor_profile(config: dict[str, Any], report: dict[str, Any]) -> None:
    profile = report.get("sensor_profile", report)
    sensors = config.setdefault("sensors", {})
    sensors["include_noisy_state"] = True
    sensors["include_imu"] = True
    sensors["state_noise_std"] = float(profile.get("state_noise_std_m", 0.0) or 0.0)
    sensors["imu_noise_std"] = float(profile.get("body_rate_noise_std_rad_s", 0.0) or 0.0)
    sensors["range_noise_std_m"] = float(profile.get("range_noise_std_m", 0.0) or 0.0)
    sensors["range_dropout_prob"] = float(profile.get("range_dropout_prob", 0.0) or 0.0)
    sensors["action_lag_s"] = float(profile.get("action_lag_s", 0.0) or 0.0)


def sim2real_section(
    evidence: dict[str, Any],
    parameter_sources: dict[str, str],
    physics: dict[str, Any] | None,
    motor: dict[str, Any] | None,
    system_id: dict[str, Any] | None,
    sensors: dict[str, Any] | None,
) -> dict[str, Any]:
    response = (system_id or {}).get("response", {})
    best = (physics or {}).get("best", {}).get("physics_profile", {})
    motor_summary = (motor or {}).get("summary", {})
    return {
        "measured": True,
        "source": "offline_composed_measured_profile",
        "confidence": "mixed_log_calibrated_and_base_priors",
        "evidence_files": [value for key, value in evidence.items() if key != "base_config"],
        "base_config": evidence["base_config"],
        "base_prior_parameters": sorted(key for key, source in parameter_sources.items() if source == "base_config"),
        "physics_linear_drag": best.get("linear_drag"),
        "physics_rate_tau_s": best.get("rate_tau_s"),
        "physics_motor_tau_s": best.get("motor_tau_s"),
        "motor_gain_imbalance": motor_summary.get("gain_imbalance"),
        "response_lag_s": quantile(response, "lag_s", "median"),
        "response_tau_s": quantile(response, "tau_s", "median"),
        "response_gain": quantile(response, "gain", "median"),
        "sensor_profile_present": sensors is not None,
    }


def quantile(report: dict[str, Any], key: str, statistic: str) -> float | None:
    value = report.get(key, {}).get(statistic)
    return float(value) if value is not None else None


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Measured Hardware Config",
        "",
        f"- Output config: `{report['output_config']}`",
        f"- Base config: `{report['base_config']}`",
        f"- Confidence: `{report['sim2real']['confidence']}`",
        "",
        "## Evidence",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in report["evidence"].items())
    lines.extend(["", "## Parameter Sources", ""])
    lines.extend(f"- `{key}`: `{value}`" for key, value in sorted(report["parameter_sources"].items()))
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
