from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import cos, hypot, radians, sin, sqrt
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class VisualWaypointConfig:
    distance_m: float = 0.30
    height_m: float = 0.55
    base_speed_m_s: float = 0.06
    policy_speed_m_s: float = 0.08
    policy_scale: float = 0.60
    policy_blend: float = 0.40
    max_lateral_residual_m_s: float = 0.02
    max_residual_step_m_s: float = 0.006
    max_total_speed_m_s: float = 0.09
    target_radius_m: float = 0.05
    max_cross_track_m: float = 0.18
    max_displacement_m: float = 0.45

    def __post_init__(self) -> None:
        if not 0.10 <= self.distance_m <= 1.00:
            raise ValueError("distance_m must be in [0.10, 1.00]")
        if not 0.20 <= self.height_m <= 0.80:
            raise ValueError("height_m must be in [0.20, 0.80]")
        if not 0.0 < self.base_speed_m_s <= 0.35:
            raise ValueError("base_speed_m_s must be in (0, 0.35]")
        if not 0.0 <= self.policy_blend <= 1.0:
            raise ValueError("policy_blend must be in [0, 1]")
        if not 0.0 < self.max_lateral_residual_m_s <= 0.048:
            raise ValueError("max_lateral_residual_m_s must be in (0, 0.048]")
        if not self.base_speed_m_s <= self.max_total_speed_m_s <= 0.38:
            raise ValueError(
                "max_total_speed_m_s must cover base_speed_m_s and be at most 0.38"
            )
        if not self.distance_m < self.max_displacement_m <= 1.20:
            raise ValueError(
                "max_displacement_m must exceed distance_m and be at most 1.20"
            )

    @property
    def policy_authority_m_s(self) -> float:
        requested = self.policy_speed_m_s * self.policy_scale * self.policy_blend
        return min(requested, self.max_lateral_residual_m_s)


@dataclass(frozen=True, slots=True)
class StraightWaypoint:
    origin_x_m: float
    origin_y_m: float
    target_x_m: float
    target_y_m: float
    target_z_m: float
    target_yaw_deg: float

    @classmethod
    def from_pose(
        cls,
        x_m: float,
        y_m: float,
        yaw_deg: float,
        config: VisualWaypointConfig,
    ) -> StraightWaypoint:
        yaw = radians(yaw_deg)
        return cls(
            origin_x_m=x_m,
            origin_y_m=y_m,
            target_x_m=x_m + config.distance_m * cos(yaw),
            target_y_m=y_m + config.distance_m * sin(yaw),
            target_z_m=config.height_m,
            target_yaw_deg=yaw_deg,
        )


@dataclass(frozen=True, slots=True)
class VisualWaypointCommand:
    vx_body_m_s: float
    vy_body_m_s: float
    policy_vy_m_s: float
    target_distance_m: float
    progress_m: float
    cross_track_m: float


def goal_intent(
    position_xyz_m: tuple[float, float, float],
    yaw_deg: float,
    waypoint: StraightWaypoint,
) -> np.ndarray:
    body_x, body_y, dz, distance = _body_goal(position_xyz_m, yaw_deg, waypoint)
    inverse = 1.0 / distance if distance > 1.0e-6 else 0.0
    yaw_error = radians(waypoint.target_yaw_deg - yaw_deg)
    return np.asarray(
        (
            body_x * inverse,
            body_y * inverse,
            dz * inverse,
            min(distance / 4.0, 1.0),
            sin(yaw_error),
            cos(yaw_error),
        ),
        dtype=np.float32,
    )


def bounded_waypoint_command(
    position_xyz_m: tuple[float, float, float],
    yaw_deg: float,
    waypoint: StraightWaypoint,
    action_vy: float,
    previous_residual_m_s: float,
    config: VisualWaypointConfig,
) -> VisualWaypointCommand:
    body_x, body_y, _dz, distance = _body_goal(
        position_xyz_m,
        yaw_deg,
        waypoint,
    )
    horizontal_distance = hypot(body_x, body_y)
    base_speed = min(
        config.base_speed_m_s,
        1.5 * max(horizontal_distance - config.target_radius_m, 0.0),
    )
    if horizontal_distance > 1.0e-6:
        base_vx = base_speed * body_x / horizontal_distance
        base_vy = base_speed * body_y / horizontal_distance
    else:
        base_vx = 0.0
        base_vy = 0.0

    requested = float(np.clip(action_vy, -1.0, 1.0))
    requested *= config.policy_authority_m_s
    residual = float(
        np.clip(
            requested,
            previous_residual_m_s - config.max_residual_step_m_s,
            previous_residual_m_s + config.max_residual_step_m_s,
        )
    )
    vx = base_vx
    vy = base_vy + residual
    speed = hypot(vx, vy)
    if speed > config.max_total_speed_m_s:
        scale = config.max_total_speed_m_s / speed
        vx *= scale
        vy *= scale

    progress, cross_track = path_coordinates(position_xyz_m, waypoint)
    return VisualWaypointCommand(
        vx_body_m_s=vx,
        vy_body_m_s=vy,
        policy_vy_m_s=residual,
        target_distance_m=distance,
        progress_m=progress,
        cross_track_m=cross_track,
    )


def path_coordinates(
    position_xyz_m: tuple[float, float, float],
    waypoint: StraightWaypoint,
) -> tuple[float, float]:
    dx = position_xyz_m[0] - waypoint.origin_x_m
    dy = position_xyz_m[1] - waypoint.origin_y_m
    yaw = radians(waypoint.target_yaw_deg)
    return (
        cos(yaw) * dx + sin(yaw) * dy,
        -sin(yaw) * dx + cos(yaw) * dy,
    )


def waypoint_envelope_abort_reason(
    position_xyz_m: tuple[float, float, float],
    waypoint: StraightWaypoint,
    config: VisualWaypointConfig,
) -> str | None:
    progress, cross_track = path_coordinates(position_xyz_m, waypoint)
    displacement = hypot(
        position_xyz_m[0] - waypoint.origin_x_m,
        position_xyz_m[1] - waypoint.origin_y_m,
    )
    if abs(cross_track) > config.max_cross_track_m:
        return f"cross_track_gt_{config.max_cross_track_m:.2f}m:{cross_track:.3f}"
    if displacement > config.max_displacement_m:
        return f"displacement_gt_{config.max_displacement_m:.2f}m:{displacement:.3f}"
    if progress < -0.10:
        return f"reverse_progress_gt_0.10m:{progress:.3f}"
    return None


def require_visual_live_readiness(
    checkpoint: str | Path,
    training_report: str | Path,
    shadow_report: str | Path,
) -> Mapping[str, object]:
    checkpoint_path = Path(checkpoint)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    training = json.loads(Path(training_report).read_text())
    shadow = json.loads(Path(shadow_report).read_text())
    if training.get("checkpoint_sha256") != digest:
        raise ValueError("training report checkpoint hash does not match")
    if not training.get("simulation_gate", {}).get("passed", False):
        raise ValueError("training report simulation gate did not pass")
    if shadow.get("checkpoint_sha256") != digest:
        raise ValueError("shadow report checkpoint hash does not match")
    if not shadow.get("next_live_shadow_gate_passed", False):
        raise ValueError("stationary live shadow gate did not pass")
    if shadow.get("controls_drone") is not False:
        raise ValueError("shadow report must be non-actuating")
    return {
        "checkpoint_sha256": digest,
        "simulation_gate_passed": True,
        "stationary_shadow_gate_passed": True,
    }


def _body_goal(
    position_xyz_m: tuple[float, float, float],
    yaw_deg: float,
    waypoint: StraightWaypoint,
) -> tuple[float, float, float, float]:
    dx = waypoint.target_x_m - position_xyz_m[0]
    dy = waypoint.target_y_m - position_xyz_m[1]
    dz = waypoint.target_z_m - position_xyz_m[2]
    yaw = radians(yaw_deg)
    body_x = cos(yaw) * dx + sin(yaw) * dy
    body_y = -sin(yaw) * dx + cos(yaw) * dy
    return body_x, body_y, dz, sqrt(dx * dx + dy * dy + dz * dz)
