from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circle import circle_tangent_yaw_from_arrays
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
    target_radius_range: tuple[float, float] | None = None


RESET_PROFILES = {
    "broad": ResetProfile("broad", 0.8, 1.0, (0.35, 0.9), (0.45, 0.9), 0.08),
    "position_yaw_easy": ResetProfile("position_yaw_easy", 0.35, 0.35, (0.45, 0.75), (0.45, 0.75), 0.04, 0.18, 0.06, 0.35),
    "position_yaw_medium": ResetProfile("position_yaw_medium", 0.65, 0.75, (0.4, 0.85), (0.4, 0.9), 0.06, 0.40, 0.12, 0.90),
    "position_yaw_wide": ResetProfile("position_yaw_wide", 0.75, 0.9, (0.38, 0.9), (0.4, 0.95), 0.07, 0.70, 0.18, 1.60),
    "position_yaw_recovery": ResetProfile("position_yaw_recovery", 0.75, 0.9, (0.35, 0.9), (0.4, 0.95), 0.16, 0.85, 0.25, 1.80),
    "position_yaw_hard": ResetProfile("position_yaw_hard", 0.8, 1.0, (0.35, 0.9), (0.4, 0.95), 0.08, 1.00, 0.25, float(np.pi)),
    "circle_easy": ResetProfile("circle_easy", 0.6, 0.6, (0.5, 0.8), (0.55, 0.75), 0.04, target_yaw_offset_abs=0.20, target_radius_range=(0.65, 0.85)),
    "circle_recovery": ResetProfile("circle_recovery", 0.9, 0.8, (0.4, 0.9), (0.5, 0.85), 0.10, target_yaw_offset_abs=0.55, target_radius_range=(0.45, 1.05)),
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
    position[:, 0] = clip_axis(position[:, 0], room.x_min, room.x_max, margin=0.25)
    position[:, 1] = clip_axis(position[:, 1], room.y_min, room.y_max, margin=0.25)
    position[:, 2] = clip_axis(position[:, 2], room.z_min, room.z_max, margin=0.25)

    yaw = rng.uniform(-np.pi, np.pi, count).astype(np.float32)
    roll = rng.normal(0.0, profile.attitude_std, count).astype(np.float32)
    pitch = rng.normal(0.0, profile.attitude_std, count).astype(np.float32)
    target = sample_target(profile, rng, position, room)
    yaw = sample_initial_yaw(profile, rng, position, target, yaw).astype(np.float32)
    target_yaw = sample_target_yaw(profile, rng, yaw).astype(np.float32)
    return position, roll, pitch, yaw, target, target_yaw


def sample_target(profile: ResetProfile, rng: np.random.Generator, position: np.ndarray, room: BoxRoom) -> np.ndarray:
    target = np.zeros_like(position)
    if profile.target_radius_range is not None:
        angle = rng.uniform(-np.pi, np.pi, len(position))
        radius = rng.uniform(*profile.target_radius_range, len(position))
        target[:, 0] = position[:, 0] - radius * np.cos(angle)
        target[:, 1] = position[:, 1] - radius * np.sin(angle)
        target[:, 2] = rng.uniform(*profile.target_z_range, len(position))
    elif profile.target_xy_offset_abs is None:
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


def clip_axis(values: np.ndarray, low: float, high: float, *, margin: float) -> np.ndarray:
    if low + margin > high - margin:
        return np.full_like(values, 0.5 * (low + high))
    return np.clip(values, low + margin, high - margin)


def sample_target_yaw(profile: ResetProfile, rng: np.random.Generator, yaw: np.ndarray) -> np.ndarray:
    if profile.target_yaw_offset_abs is None:
        return rng.uniform(-np.pi, np.pi, len(yaw))
    return wrap_angle(yaw + rng.uniform(-profile.target_yaw_offset_abs, profile.target_yaw_offset_abs, len(yaw)))


def sample_initial_yaw(
    profile: ResetProfile,
    rng: np.random.Generator,
    position: np.ndarray,
    target: np.ndarray,
    yaw: np.ndarray,
) -> np.ndarray:
    if profile.target_radius_range is None or profile.target_yaw_offset_abs is None:
        return yaw
    tangent_yaw = circle_tangent_yaw_from_arrays(position, target)
    offset = rng.uniform(-profile.target_yaw_offset_abs, profile.target_yaw_offset_abs, len(yaw))
    return wrap_angle(tangent_yaw + offset)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
