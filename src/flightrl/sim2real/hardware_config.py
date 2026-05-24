from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


DYNAMICS_KEYS = [
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
]


def summarize_hardware_model(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "source": None, "measured": False, "parameters": {}}
    data = tomllib.loads(path.read_text())
    sim2real = data.get("sim2real", {})
    parameters = {key: data.get("drone", {}).get(key) for key in DYNAMICS_KEYS}
    missing = [key for key, value in parameters.items() if value is None]
    measured = bool(sim2real.get("measured", False)) and not missing
    return {
        "present": True,
        "path": str(path),
        "source": sim2real.get("source") or filename_source(path),
        "measured": measured,
        "missing_parameters": missing,
        "dt": data.get("environment", {}).get("dt"),
        "action_mode": data.get("environment", {}).get("action_mode"),
        "parameters": parameters,
        "domain_randomization": data.get("domain_randomization", {}),
        "sensor_model": data.get("sensors", {}),
    }


def filename_source(path: Path) -> str:
    if "placeholder" in path.name:
        return "placeholder"
    return "unspecified"
