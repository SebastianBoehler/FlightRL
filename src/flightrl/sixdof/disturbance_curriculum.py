from __future__ import annotations

import argparse
from typing import Any

from .disturbance import SixDofDisturbanceProfile, configure_disturbance, resolve_disturbance_profile


def add_disturbance_curriculum_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--disturbance-ramp-start-profile", default=None)
    parser.add_argument("--disturbance-ramp-updates", type=int, default=0)


def configure_training_disturbance(env, args: argparse.Namespace, *, update: int, total_updates: int) -> SixDofDisturbanceProfile:
    profile = training_disturbance_profile(args, update=update, total_updates=total_updates)
    if getattr(env, "use_native_step", False) and profile.enabled:
        raise ValueError("disturbance profiles are not yet applied by the native 6-DoF step")
    return configure_disturbance(env, profile)


def training_disturbance_profile(args: argparse.Namespace, *, update: int, total_updates: int) -> SixDofDisturbanceProfile:
    end = resolve_disturbance_profile(args.disturbance_profile)
    start_spec = getattr(args, "disturbance_ramp_start_profile", None)
    if not start_spec:
        return end
    start = resolve_disturbance_profile(start_spec)
    return interpolate_disturbance_profile(start, end, ramp_fraction(update, total_updates, getattr(args, "disturbance_ramp_updates", 0)))


def ramp_fraction(update: int, total_updates: int, ramp_updates: int = 0) -> float:
    steps = int(ramp_updates) if int(ramp_updates) > 0 else int(total_updates)
    if steps <= 1:
        return 1.0
    clamped = min(max(int(update), 1), steps)
    return float((clamped - 1) / (steps - 1))


def interpolate_disturbance_profile(start: SixDofDisturbanceProfile, end: SixDofDisturbanceProfile, fraction: float) -> SixDofDisturbanceProfile:
    frac = min(max(float(fraction), 0.0), 1.0)
    return SixDofDisturbanceProfile(
        name=f"{start.name}_to_{end.name}_{frac:.2f}",
        world_accel_xy_m_s2=lerp_pair(start.world_accel_xy_m_s2, end.world_accel_xy_m_s2, frac),
        world_accel_z_m_s2=lerp_pair(start.world_accel_z_m_s2, end.world_accel_z_m_s2, frac),
    )


def lerp_pair(start: tuple[float, float], end: tuple[float, float], fraction: float) -> tuple[float, float]:
    return (float(start[0] + (end[0] - start[0]) * fraction), float(start[1] + (end[1] - start[1]) * fraction))


def disturbance_curriculum_context(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": bool(getattr(args, "disturbance_ramp_start_profile", None)),
        "start_profile": getattr(args, "disturbance_ramp_start_profile", None),
        "end_profile": args.disturbance_profile,
        "ramp_updates": int(getattr(args, "disturbance_ramp_updates", 0)),
    }
