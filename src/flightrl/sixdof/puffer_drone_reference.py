from __future__ import annotations

import configparser
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .env import ACTION_DIM, OBSERVATION_DIM, TASK_IDS
from .motor_rpm import resolve_motor_rpm_params
from .physics import resolve_domain_randomization, resolve_physics_profile


MACRO_NAMES = (
    "BASE_MASS",
    "BASE_IXX",
    "BASE_IYY",
    "BASE_IZZ",
    "BASE_ARM_LEN",
    "BASE_K_THRUST",
    "BASE_K_DRAG",
    "BASE_GRAVITY",
    "BASE_MAX_RPM",
    "BASE_K_MOT",
    "BASE_MAX_VEL",
    "BASE_MAX_OMEGA",
    "DRONE_OBS_SIZE",
    "DT",
    "ACTION_SUBSTEPS",
)


def build_reference_report(pufferlib_root: Path) -> dict[str, Any]:
    root = Path(pufferlib_root)
    official = official_contract(root)
    local = flightrl_contract()
    return {
        "pufferlib_root": str(root),
        "official_puffer_drone": official,
        "flightrl": local,
        "compatibility": compatibility(official, local),
        "recommendation": {
            "decision": "use_official_puffer_drone_as_speed_baseline_keep_flightrl_as_transfer_env",
            "next_steps": [
                "Run official puffer drone unchanged and record wall-clock/SPS baseline.",
                "Use FlightRL physics_profile='puffer_drone' with action_mode='motor_rpm' for official-style benchmark runs.",
                "Port renderer ideas after the contract report stays stable.",
                "Keep live deployment blocked by existing replay, safety, and transfer gates.",
            ],
        },
    }


def official_contract(root: Path) -> dict[str, Any]:
    drone_dir = root / "ocean" / "drone"
    macros = parse_macros(drone_dir / "dronelib.h", MACRO_NAMES)
    binding = parse_macros(drone_dir / "binding.c", ("NUM_ATNS",))
    config = parse_drone_config(root / "config" / "drone.ini")
    hover_rpm = math.sqrt((macros["BASE_MASS"] * macros["BASE_GRAVITY"]) / (4.0 * macros["BASE_K_THRUST"]))
    min_rpm = max(0.0, 2.0 * hover_rpm - macros["BASE_MAX_RPM"])
    return {
        "source": "PufferLib ocean/drone",
        "observation_dim": int(macros["DRONE_OBS_SIZE"]),
        "action_dim": int(binding.get("NUM_ATNS", 4)),
        "action_semantics": "four normalized per-motor commands mapped to RPM targets",
        "control_rate_hz": 1.0 / (macros["DT"] * macros["ACTION_SUBSTEPS"]),
        "physics_step_hz": 1.0 / macros["DT"],
        "hover_rpm": hover_rpm,
        "min_centered_hover_rpm": min_rpm,
        "constants": macros,
        "config": config,
        "tasks": ["hover", "race", "sphere", "cube", "flag"],
        "renderer": "raylib 3D scene with all env num_agents, trails, target geometry, and motor RPM HUD",
    }


def flightrl_contract() -> dict[str, Any]:
    physics = resolve_physics_profile("puffer_drone")
    randomization = resolve_domain_randomization(None)
    motor = resolve_motor_rpm_params("puffer_drone")
    min_rpm = max(0.0, 2.0 * motor.hover_rpm - motor.max_rpm)
    return {
        "source": "FlightRL SixDofCrazyflieEnv puffer_drone motor_rpm lane",
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "action_modes": {
            "body_rate": "collective thrust plus roll/pitch/yaw body-rate commands",
            "motor_rpm": "four normalized per-motor commands mapped to RPM targets",
        },
        "control_rate_hz": 100.0,
        "physics_step_hz": 100.0 * motor.physics_substeps,
        "hover_rpm": motor.hover_rpm,
        "min_centered_hover_rpm": min_rpm,
        "motor_rpm_params": asdict(motor),
        "physics_profile": asdict(physics),
        "domain_randomization": asdict(randomization),
        "tasks": sorted(TASK_IDS),
        "renderer": "Python/planar renderer plus room-log visualizers; exported Puffer 6-DoF render is currently stubbed",
    }


def compatibility(official: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    obs_match = official["observation_dim"] == local["observation_dim"]
    action_match = official["action_dim"] == local["action_dim"]
    cadence_match = abs(official["control_rate_hz"] - local["control_rate_hz"]) < 1e-6
    motor_mode = local["action_modes"]["motor_rpm"]
    blockers = []
    if not obs_match:
        blockers.append("observation_contract_differs")
    if not action_match:
        blockers.append("action_dim_differs")
    if "motor" not in motor_mode:
        blockers.append("no_motor_rpm_action_mode")
    if official["constants"]["BASE_MASS"] != local["physics_profile"]["mass_kg"]:
        blockers.append("mass_profile_differs")
    if official["constants"]["BASE_K_MOT"] != local["motor_rpm_params"]["motor_tau_s"]:
        blockers.append("motor_time_constant_differs")
    return {
        "action_dim_match": action_match,
        "observation_dim_match": obs_match,
        "control_rate_match": cadence_match,
        "has_flightrl_motor_rpm_mode": "motor" in motor_mode,
        "official_hover_rpm": official["hover_rpm"],
        "flightrl_hover_rpm": local["hover_rpm"],
        "adaptation_required_for_replacement": bool(blockers),
        "replacement_blockers": blockers,
        "safe_use": "baseline_and_alignment_reference",
    }


def parse_macros(path: Path, names: tuple[str, ...]) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"missing Puffer drone source file: {path}")
    wanted = set(names)
    values: dict[str, float] = {}
    for line in path.read_text().splitlines():
        match = re.match(r"\s*#define\s+([A-Z0-9_]+)\s+(.+?)(?:\s*//.*)?$", line)
        if not match or match.group(1) not in wanted:
            continue
        values[match.group(1)] = parse_number(match.group(2))
    missing = wanted - values.keys()
    if missing:
        raise ValueError(f"missing macros in {path}: {', '.join(sorted(missing))}")
    return values


def parse_number(raw: str) -> float:
    text = raw.strip().rstrip("f")
    if text.startswith("("):
        raise ValueError(f"unsupported macro expression: {raw}")
    return float(text)


def parse_drone_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing Puffer drone config: {path}")
    parser = configparser.ConfigParser()
    parser.read(path)
    return {
        "vec": {key: number_or_text(value) for key, value in parser["vec"].items()},
        "env": {key: number_or_text(value) for key, value in parser["env"].items()},
        "policy": {key: number_or_text(value) for key, value in parser["policy"].items()},
    }


def number_or_text(value: str) -> int | float | str:
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def render_markdown(report: dict[str, Any]) -> str:
    official = report["official_puffer_drone"]
    local = report["flightrl"]
    comp = report["compatibility"]
    blockers = ", ".join(comp["replacement_blockers"]) or "none"
    lines = [
        "# Puffer Drone Reference Alignment",
        "",
        f"- PufferLib root: `{report['pufferlib_root']}`",
        f"- Decision: `{report['recommendation']['decision']}`",
        f"- Replacement requires adaptation: `{comp['adaptation_required_for_replacement']}`",
        f"- Replacement blockers: `{blockers}`",
        "",
        "| contract | official Puffer drone | FlightRL |",
        "| --- | --- | --- |",
        f"| observation dim | {official['observation_dim']} | {local['observation_dim']} |",
        f"| action dim | {official['action_dim']} | {local['action_dim']} |",
        f"| action model | {official['action_semantics']} | {local['action_modes']['motor_rpm']} |",
        f"| control rate | {official['control_rate_hz']:.1f} Hz | {local['control_rate_hz']:.1f} Hz |",
        f"| physics step | {official['physics_step_hz']:.1f} Hz | {local['physics_step_hz']:.1f} Hz |",
        f"| mass | {official['constants']['BASE_MASS']:.3f} kg | {local['physics_profile']['mass_kg']:.3f} kg |",
        f"| motor tau | {official['constants']['BASE_K_MOT']:.3f} s | {local['motor_rpm_params']['motor_tau_s']:.3f} s |",
        f"| hover RPM | {official['hover_rpm']:.0f} | {local['hover_rpm']:.0f} |",
        "",
        "## Next Steps",
        "",
    ]
    lines.extend(f"- {item}" for item in report["recommendation"]["next_steps"])
    return "\n".join(lines)
