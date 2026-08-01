from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from flightrl.evidence_values import failure_strings, finite_number
from flightrl.sim2real.audit_evidence import valid_hardware_parameters


DRONE_KEYS = {
    "mass",
    "inertia",
    "arm_length",
    "drag",
    "angular_drag",
    "hover_thrust",
    "thrust_gain",
    "max_total_thrust",
    "max_pitch_torque",
    "actuator_tau",
}
SENSOR_KEYS = {"state_noise_std", "imu_noise_std"}
SENSOR_REQUIRED_KEYS = {
    *SENSOR_KEYS,
    "attitude_noise_std_deg",
    "command_latency_s",
    "range_noise_std_mm",
}
RANDOMIZATION_KEYS = {"enabled", "mass_scale", "drag_scale", "thrust_scale", "actuator_tau_scale", "sensor_noise_scale"}


def export_config(profile_path: Path, *, base_config: Path, output_config: Path) -> dict[str, Any]:
    raw_profile = json.loads(profile_path.read_text())
    profile = raw_profile if isinstance(raw_profile, dict) else {}
    report = {
        "profile": str(profile_path),
        "base_config": str(base_config),
        "output_config": str(output_config),
        "exported": False,
        "failures": [],
        "safety": "Exported config is simulator-only and still requires readiness/replay gates before live use.",
    }
    summary = profile.get("summary", {})
    typed_failures = failure_strings(summary.get("failures")) if isinstance(summary, dict) else None
    if not isinstance(summary, dict) or summary.get("profile_ready") is not True or typed_failures != []:
        details = typed_failures if typed_failures is not None else ["profile_failures_invalid"]
        report["failures"] = ["profile_not_ready", *details]
        return report
    overlay = profile.get("simulator_overlay")
    overlay_failures = validate_overlay(overlay)
    if overlay_failures:
        report["failures"] = overlay_failures
        return report
    raw = tomllib.loads(base_config.read_text())
    merged = apply_overlay(raw, overlay, profile_path)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(render_toml(merged) + "\n")
    report["exported"] = True
    report["sections"] = sorted(merged)
    return report


def validate_overlay(overlay: object) -> list[str]:
    if overlay is None:
        return ["simulator_overlay_missing"]
    if not isinstance(overlay, dict):
        return ["simulator_overlay_invalid"]
    drone = overlay.get("drone")
    if not valid_hardware_parameters(drone):
        return ["simulator_overlay_drone_invalid"]
    failures = []
    for section in ("actuator", "sensors", "domain_randomization"):
        if not isinstance(overlay.get(section), dict):
            failures.append(f"simulator_overlay_{section}_invalid")
    if failures:
        return failures
    sensors = overlay["sensors"]
    if set(sensors) != SENSOR_REQUIRED_KEYS or not nonnegative_values(sensors.values()):
        failures.append("simulator_overlay_sensors_invalid")
    randomization = overlay["domain_randomization"]
    if type(randomization.get("enabled")) is not bool:
        failures.append("simulator_overlay_domain_randomization_invalid")
    if not nonnegative_values(value for key, value in randomization.items() if key != "enabled"):
        failures.append("simulator_overlay_domain_randomization_invalid")
    actuator = overlay["actuator"]
    gains = actuator.get("relative_motor_gains")
    actuator_values = (value for key, value in actuator.items() if key not in {"present", "relative_motor_gains"})
    if (
        actuator.get("present") is not True
        or not nonnegative_values(actuator_values)
        or not isinstance(gains, dict)
        or set(gains) != {"1", "2", "3", "4"}
        or not positive_values(gains.values())
    ):
        failures.append("simulator_overlay_actuator_invalid")
    return sorted(dict.fromkeys(failures))


def nonnegative_values(values) -> bool:
    parsed = [finite_number(value) for value in values]
    return all(value is not None and value >= 0.0 for value in parsed)


def positive_values(values) -> bool:
    parsed = [finite_number(value) for value in values]
    return all(value is not None and value > 0.0 for value in parsed)


def apply_overlay(raw: dict[str, Any], overlay: dict[str, Any], profile_path: Path) -> dict[str, Any]:
    merged = json.loads(json.dumps(raw))
    drone = merged.setdefault("drone", {})
    for key, value in overlay.get("drone", {}).items():
        if key in DRONE_KEYS and value is not None:
            drone[key] = value
    sensors = merged.setdefault("sensors", {})
    sensor_overlay = overlay.get("sensors", {})
    for key in SENSOR_KEYS:
        if sensor_overlay.get(key) is not None:
            sensors[key] = sensor_overlay[key]
    if sensor_overlay:
        sensors["include_noisy_state"] = True
        sensors["include_imu"] = True
    randomization = merged.setdefault("domain_randomization", {})
    for key, value in overlay.get("domain_randomization", {}).items():
        if key in RANDOMIZATION_KEYS and value is not None:
            randomization[key] = value
    merged["sim2real"] = {
        "profile": str(profile_path),
        "actuator_priors_present": bool(overlay.get("actuator", {}).get("present", False)),
        "command_latency_s": sensor_overlay.get("command_latency_s"),
        "range_noise_std_mm": sensor_overlay.get("range_noise_std_mm"),
    }
    return merged


def render_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {toml_value(value)}")
    return "\n".join(lines)


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if value is None:
        return '""'
    raise TypeError(f"unsupported TOML value {value!r}")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sim-To-Real Config Export",
        "",
        f"- Profile: `{report['profile']}`",
        f"- Base config: `{report['base_config']}`",
        f"- Output config: `{report['output_config']}`",
        f"- Exported: `{report['exported']}`",
        f"- Failures: `{', '.join(report.get('failures', [])) or 'none'}`",
        "",
        report["safety"],
    ]
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
