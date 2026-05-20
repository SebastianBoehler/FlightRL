from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import BoxRoom


@dataclass(frozen=True, slots=True)
class ResetProfile:
    name: str
    initial_xy_abs: float
    target_xy_abs: float
    z_range: tuple[float, float]
    target_z_range: tuple[float, float]
    attitude_std: float
    target_xy_offset_abs: float | None = None
    target_z_offset_abs: float | None = None
    target_yaw_offset_abs: float | None = None


RESET_PROFILES = {
    "broad": ResetProfile("broad", 0.8, 1.0, (0.35, 0.9), (0.45, 0.9), 0.08),
    "position_yaw_easy": ResetProfile("position_yaw_easy", 0.35, 0.35, (0.45, 0.75), (0.45, 0.75), 0.04, 0.18, 0.06, 0.35),
    "position_yaw_medium": ResetProfile("position_yaw_medium", 0.65, 0.75, (0.4, 0.85), (0.4, 0.9), 0.06, 0.40, 0.12, 0.90),
}


def resolve_reset_profile(value: str | ResetProfile | None) -> ResetProfile:
    if isinstance(value, ResetProfile):
        return value
    name = value or "broad"
    if name not in RESET_PROFILES:
        raise ValueError(f"unknown 6-DoF reset profile {name!r}; expected one of {', '.join(sorted(RESET_PROFILES))}")
    return RESET_PROFILES[name]


def sample_reset(
    profile: ResetProfile,
    rng: np.random.Generator,
    count: int,
    room: BoxRoom,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = np.zeros((count, 3), dtype=np.float32)
    position[:, 0] = rng.uniform(-profile.initial_xy_abs, profile.initial_xy_abs, count)
    position[:, 1] = rng.uniform(-profile.initial_xy_abs, profile.initial_xy_abs, count)
    position[:, 2] = rng.uniform(*profile.z_range, count)

    yaw = rng.uniform(-np.pi, np.pi, count).astype(np.float32)
    roll = rng.normal(0.0, profile.attitude_std, count).astype(np.float32)
    pitch = rng.normal(0.0, profile.attitude_std, count).astype(np.float32)
    target = sample_target(profile, rng, position, room)
    target_yaw = sample_target_yaw(profile, rng, yaw).astype(np.float32)
    return position, roll, pitch, yaw, target, target_yaw


def sample_target(profile: ResetProfile, rng: np.random.Generator, position: np.ndarray, room: BoxRoom) -> np.ndarray:
    target = np.zeros_like(position)
    if profile.target_xy_offset_abs is None:
        target[:, 0] = rng.uniform(-profile.target_xy_abs, profile.target_xy_abs, len(position))
        target[:, 1] = rng.uniform(-profile.target_xy_abs, profile.target_xy_abs, len(position))
        target[:, 2] = rng.uniform(*profile.target_z_range, len(position))
    else:
        target[:, 0] = position[:, 0] + rng.uniform(-profile.target_xy_offset_abs, profile.target_xy_offset_abs, len(position))
        target[:, 1] = position[:, 1] + rng.uniform(-profile.target_xy_offset_abs, profile.target_xy_offset_abs, len(position))
        target[:, 2] = position[:, 2] + rng.uniform(-profile.target_z_offset_abs, profile.target_z_offset_abs, len(position))
    target[:, 0] = np.clip(target[:, 0], room.x_min + 0.25, room.x_max - 0.25)
    target[:, 1] = np.clip(target[:, 1], room.y_min + 0.25, room.y_max - 0.25)
    target[:, 2] = np.clip(target[:, 2], room.z_min + 0.35, room.z_max - 0.25)
    return target.astype(np.float32)


def sample_target_yaw(profile: ResetProfile, rng: np.random.Generator, yaw: np.ndarray) -> np.ndarray:
    if profile.target_yaw_offset_abs is None:
        return rng.uniform(-np.pi, np.pi, len(yaw))
    return wrap_angle(yaw + rng.uniform(-profile.target_yaw_offset_abs, profile.target_yaw_offset_abs, len(yaw)))


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
