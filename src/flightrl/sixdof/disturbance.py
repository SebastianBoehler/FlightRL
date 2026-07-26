from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SixDofDisturbanceProfile:
    name: str
    world_accel_xy_m_s2: tuple[float, float] = (0.0, 0.0)
    world_accel_z_m_s2: tuple[float, float] = (0.0, 0.0)

    @property
    def enabled(self) -> bool:
        return self.world_accel_xy_m_s2 != (0.0, 0.0) or self.world_accel_z_m_s2 != (0.0, 0.0)


NO_DISTURBANCE = SixDofDisturbanceProfile("none")
RAW_LIVE_DRIFT_DISTURBANCE = SixDofDisturbanceProfile(
    "raw_live_drift",
    world_accel_xy_m_s2=(0.25, 0.55),
    world_accel_z_m_s2=(-0.05, 0.05),
)
RAW_LIVE_MILD_DISTURBANCE = SixDofDisturbanceProfile(
    "raw_live_mild",
    world_accel_xy_m_s2=(0.10, 0.25),
    world_accel_z_m_s2=(-0.03, 0.03),
)


def resolve_disturbance_profile(value: str | SixDofDisturbanceProfile | None) -> SixDofDisturbanceProfile:
    if isinstance(value, SixDofDisturbanceProfile):
        return value
    if value is None or value in {"none", "off", "disabled"}:
        return NO_DISTURBANCE
    if value == "raw_live_drift":
        return RAW_LIVE_DRIFT_DISTURBANCE
    if value == "raw_live_mild":
        return RAW_LIVE_MILD_DISTURBANCE
    path = Path(value)
    if path.exists():
        return disturbance_profile_from_json(path)
    raise ValueError(f"unknown 6-DoF disturbance profile {value!r}")


def disturbance_profile_from_json(path: str | Path) -> SixDofDisturbanceProfile:
    payload = json.loads(Path(path).read_text())
    data = payload.get("disturbance_profile", payload)
    if not isinstance(data, dict):
        raise ValueError(f"disturbance profile in {path} must be an object")
    return disturbance_profile_from_dict(data)


def disturbance_profile_from_dict(data: dict[str, Any]) -> SixDofDisturbanceProfile:
    if "name" not in data:
        raise ValueError("disturbance profile is missing name")
    return SixDofDisturbanceProfile(
        name=str(data["name"]),
        world_accel_xy_m_s2=profile_pair(data, "world_accel_xy_m_s2"),
        world_accel_z_m_s2=profile_pair(data, "world_accel_z_m_s2"),
    )


def profile_pair(data: dict[str, Any], key: str) -> tuple[float, float]:
    if key not in data:
        raise ValueError(f"disturbance profile is missing {key}")
    raw = data[key]
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{key} must contain exactly two numbers")
    pair = (float(raw[0]), float(raw[1]))
    if not np.all(np.isfinite(pair)) or pair[0] > pair[1]:
        raise ValueError(f"{key} must be finite and sorted low-to-high")
    return pair


def configure_disturbance(env, value: str | SixDofDisturbanceProfile | None) -> SixDofDisturbanceProfile:
    profile = resolve_disturbance_profile(value)
    accel = np.zeros((env.num_envs, 3), dtype=np.float32)
    if profile.enabled:
        angles = env.rng.uniform(-np.pi, np.pi, env.num_envs)
        xy_mag = env.rng.uniform(*profile.world_accel_xy_m_s2, env.num_envs)
        accel[:, 0] = xy_mag * np.cos(angles)
        accel[:, 1] = xy_mag * np.sin(angles)
        accel[:, 2] = env.rng.uniform(*profile.world_accel_z_m_s2, env.num_envs)
    env.disturbance_profile = profile
    env.disturbance_world_accel = accel
    return profile


def disturbance_accel(env) -> np.ndarray | None:
    accel = getattr(env, "disturbance_world_accel", None)
    if accel is None or not np.any(accel):
        return None
    return accel
