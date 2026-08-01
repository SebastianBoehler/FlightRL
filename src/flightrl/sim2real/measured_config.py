from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from flightrl.evidence_values import exact_nonnegative_int, exact_true, failure_strings, finite_number
from flightrl.sim2real.hardware_config import DYNAMICS_KEYS
from flightrl.sim2real.profile_export import render_toml


SIM_ALIGNMENT_TO_DRONE = {
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
    physics_ready = valid_physics_sim_alignment(physics)
    if physics is not None:
        evidence["physics_sim_alignment"] = str(physics_calibration)
    if physics_ready:
        apply_physics(config, physics, parameter_sources)

    motor = read_json(motor_calibration)
    motor_ready = valid_motor_calibration(motor)
    if motor is not None:
        evidence["motor_calibration"] = str(motor_calibration)
    if motor_ready:
        apply_motor(config, motor)

    system_id = read_json(live_system_id)
    system_id_ready = valid_system_id(system_id)
    if system_id is not None:
        evidence["live_system_id"] = str(live_system_id)

    sensors = read_json(sensor_profile)
    sensors_ready = valid_summary_report(sensors, "profile_ready") and valid_sensor_profile(sensors)
    if sensors is not None:
        evidence["sensor_profile"] = str(sensor_profile)
    if sensors_ready:
        apply_sensor_profile(config, sensors)

    evidence_ready = {
        "physics_sim_alignment": physics_ready,
        "motor_calibration": motor_ready,
        "live_system_id": system_id_ready,
        "sensor_profile": sensors_ready,
    }
    config["sim2real"] = sim2real_section(
        evidence,
        evidence_ready,
        parameter_sources,
        physics if physics_ready else None,
        motor if motor_ready else None,
        system_id if system_id_ready else None,
        sensors if sensors_ready else None,
    )
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
    for source_key, drone_key in SIM_ALIGNMENT_TO_DRONE.items():
        value = profile.get(source_key)
        if value is not None:
            drone[drone_key] = value
            parameter_sources[drone_key] = "sim_alignment"
    thrust_scale = finite_number(profile.get("thrust_scale"))
    if thrust_scale is not None and thrust_scale >= 0.0:
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
    evidence_ready: dict[str, bool],
    parameter_sources: dict[str, str],
    physics: dict[str, Any] | None,
    motor: dict[str, Any] | None,
    system_id: dict[str, Any] | None,
    sensors: dict[str, Any] | None,
) -> dict[str, Any]:
    response = (system_id or {}).get("response", {})
    best = (physics or {}).get("best", {}).get("physics_profile", {})
    motor_summary = (motor or {}).get("summary", {})
    base_priors = sorted(key for key in DYNAMICS_KEYS if parameter_sources.get(key) == "base_config")
    sim_alignment = sorted(key for key in DYNAMICS_KEYS if parameter_sources.get(key) == "sim_alignment")
    missing_dynamics = sorted(key for key in DYNAMICS_KEYS if key not in parameter_sources)
    unmeasured_dynamics = sorted(key for key in DYNAMICS_KEYS if parameter_sources.get(key) != "hardware_measurement")
    failures = [f"{name}_not_ready" for name, ready in evidence_ready.items() if not ready]
    if base_priors:
        failures.append("base_prior_dynamics_present")
    if missing_dynamics:
        failures.append("required_dynamics_missing")
    if unmeasured_dynamics:
        failures.append("measured_hardware_dynamics_missing")
    measured = not failures
    return {
        "measured": measured,
        "source": "offline_composed_sim_alignment_profile",
        "confidence": "validated_measured_dynamics" if measured else "non_authoritative_composed_profile",
        "evidence_files": [value for key, value in evidence.items() if key != "base_config"],
        "measurement_failures": failures,
        "base_config": evidence["base_config"],
        "base_prior_parameters": base_priors,
        "sim_alignment_parameters": sim_alignment,
        "unmeasured_dynamics_parameters": unmeasured_dynamics,
        "missing_dynamics_parameters": missing_dynamics,
        **{f"{name}_ready": ready for name, ready in evidence_ready.items()},
        "sim_alignment_linear_drag": best.get("linear_drag"),
        "sim_alignment_rate_tau_s": best.get("rate_tau_s"),
        "sim_alignment_motor_tau_s": best.get("motor_tau_s"),
        "motor_gain_imbalance": motor_summary.get("gain_imbalance"),
        "response_lag_s": quantile(response, "lag_s", "median"),
        "response_tau_s": quantile(response, "tau_s", "median"),
        "response_gain": quantile(response, "gain", "median"),
        "sensor_profile_present": sensors is not None,
    }


def quantile(report: dict[str, Any], key: str, statistic: str) -> float | None:
    section = report.get(key, {})
    return finite_number(section.get(statistic)) if isinstance(section, dict) else None


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else None


def valid_physics_sim_alignment(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict) or not exact_true(report.get("passed")):
        return False
    failures = failure_strings(report.get("failures", []))
    best = report.get("best")
    profile = best.get("physics_profile", {}) if isinstance(best, dict) else {}
    if failures != [] or not isinstance(profile, dict):
        return False
    mass = finite_number(profile.get("mass_kg"))
    drag = finite_number(profile.get("linear_drag"))
    rate_tau = finite_number(profile.get("rate_tau_s"))
    return mass is not None and mass > 0.0 and drag is not None and drag >= 0.0 and rate_tau is not None and rate_tau >= 0.0


def valid_summary_report(report: dict[str, Any] | None, ready_flag: str) -> bool:
    if not isinstance(report, dict):
        return False
    summary = report.get("summary", {})
    return (
        isinstance(summary, dict)
        and exact_true(summary.get(ready_flag))
        and failure_strings(summary.get("failures", [])) == []
    )


def valid_motor_calibration(report: dict[str, Any] | None) -> bool:
    if not valid_summary_report(report, "passed"):
        return False
    assert report is not None
    gain_imbalance = finite_number(report["summary"].get("gain_imbalance"))
    return gain_imbalance is not None and 0.0 <= gain_imbalance <= 1.0


def valid_system_id(report: dict[str, Any] | None) -> bool:
    if not valid_summary_report(report, "profile_ready"):
        return False
    assert report is not None
    summary = report["summary"]
    tracking_runs = exact_nonnegative_int(summary.get("tracking_runs"))
    tracking_samples = exact_nonnegative_int(summary.get("tracking_samples"))
    response = report.get("response", {})
    lag = quantile(response, "lag_s", "median") if isinstance(response, dict) else None
    tau = quantile(response, "tau_s", "median") if isinstance(response, dict) else None
    gain = quantile(response, "gain", "median") if isinstance(response, dict) else None
    return (
        tracking_runs is not None
        and tracking_runs > 0
        and tracking_samples is not None
        and tracking_samples > 0
        and lag is not None
        and lag >= 0.0
        and tau is not None
        and tau >= 0.0
        and gain is not None
        and gain > 0.0
    )


def valid_sensor_profile(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    profile = report.get("sensor_profile", {})
    if not isinstance(profile, dict):
        return False
    names = (
        "state_noise_std_m",
        "body_rate_noise_std_rad_s",
        "range_noise_std_m",
        "range_dropout_prob",
        "action_lag_s",
    )
    values = {name: finite_number(profile.get(name)) for name in names}
    dropout = values["range_dropout_prob"]
    return all(value is not None and value >= 0.0 for value in values.values()) and dropout is not None and dropout <= 1.0


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
