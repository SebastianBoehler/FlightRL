from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import hypot, isfinite, pi
from typing import Any, Literal, Mapping, Sequence


MissionMetricCompatibility = Literal[
    "current",
    "incompatible",
]


@dataclass(frozen=True, slots=True)
class DoorMissionSample:
    position_m: Sequence[float]
    velocity_m_s: Sequence[float]
    yaw_rad: float
    yaw_rate_rad_s: float
    room_bounds_m: Sequence[float]
    door_face: int
    target_position_m: Sequence[float]
    target_yaw_rad: float
    visible: bool


@dataclass(frozen=True, slots=True)
class DoorMissionEvaluation:
    in_tolerance: bool
    dwell_steps: int
    success: bool


@dataclass(frozen=True, slots=True)
class DoorMissionMetric:
    metric_id: str
    schema_version: int
    target_standoff_m: float
    planar_position_tolerance_m: float
    vertical_position_tolerance_m: float
    standoff_tolerance_m: float
    yaw_alignment_tolerance_rad: float
    max_horizontal_speed_m_s: float
    max_vertical_speed_m_s: float
    max_yaw_rate_rad_s: float
    dwell_steps: int

    def __post_init__(self) -> None:
        numeric = (
            self.target_standoff_m,
            self.planar_position_tolerance_m,
            self.vertical_position_tolerance_m,
            self.standoff_tolerance_m,
            self.yaw_alignment_tolerance_rad,
            self.max_horizontal_speed_m_s,
            self.max_vertical_speed_m_s,
            self.max_yaw_rate_rad_s,
        )
        if not self.metric_id or self.schema_version != 1:
            raise ValueError("invalid fixed-door mission metric identity")
        if any(not isfinite(value) or value <= 0.0 for value in numeric):
            raise ValueError("fixed-door mission thresholds must be positive")
        if type(self.dwell_steps) is not int or self.dwell_steps <= 0:
            raise ValueError("fixed-door mission dwell must be positive")

    def payload(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "schema_version": self.schema_version,
            "target_standoff_m": self.target_standoff_m,
            "planar_position_tolerance_m": self.planar_position_tolerance_m,
            "vertical_position_tolerance_m": self.vertical_position_tolerance_m,
            "standoff_tolerance_m": self.standoff_tolerance_m,
            "yaw_alignment_tolerance_rad": self.yaw_alignment_tolerance_rad,
            "max_horizontal_speed_m_s": self.max_horizontal_speed_m_s,
            "max_vertical_speed_m_s": self.max_vertical_speed_m_s,
            "max_yaw_rate_rad_s": self.max_yaw_rate_rad_s,
            "dwell_steps": self.dwell_steps,
        }

    def sha256(self) -> str:
        encoded = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_report(self) -> dict[str, Any]:
        return self.payload() | {"sha256": self.sha256()}

    def env_values(self) -> dict[str, int | float]:
        payload = self.payload()
        return {
            f"mission_{key}": value
            for key, value in payload.items()
            if key not in {"metric_id", "schema_version"}
        }

    def verify_env(self, env: Mapping[str, Any]) -> None:
        for key, expected in self.env_values().items():
            actual = env.get(key)
            if isinstance(expected, int):
                matches = type(actual) is int and actual == expected
            else:
                matches = isinstance(actual, (int, float)) and isfinite(actual)
                matches = matches and abs(float(actual) - expected) <= 1.0e-7
            if not matches:
                raise ValueError(
                    f"fixed-door mission metric mismatch for {key}: "
                    f"expected {expected}, got {actual!r}"
                )

    def evaluate(
        self,
        sample: DoorMissionSample,
        *,
        prior_dwell_steps: int,
    ) -> DoorMissionEvaluation:
        if (
            isinstance(prior_dwell_steps, bool)
            or not isinstance(prior_dwell_steps, int)
            or prior_dwell_steps < 0
            or prior_dwell_steps > self.dwell_steps
        ):
            raise ValueError("prior mission dwell is outside the metric state range")
        valid_shape = (
            isinstance(sample.position_m, Sequence)
            and len(sample.position_m) == 3
            and isinstance(sample.velocity_m_s, Sequence)
            and len(sample.velocity_m_s) == 3
            and isinstance(sample.room_bounds_m, Sequence)
            and len(sample.room_bounds_m) == 7
            and isinstance(sample.target_position_m, Sequence)
            and len(sample.target_position_m) == 3
            and type(sample.door_face) is int
            and sample.door_face in range(4)
            and type(sample.visible) is bool
        )
        if not valid_shape:
            return DoorMissionEvaluation(False, 0, False)
        values = (
            *sample.position_m,
            *sample.velocity_m_s,
            sample.yaw_rad,
            sample.yaw_rate_rad_s,
            *sample.room_bounds_m,
            *sample.target_position_m,
            sample.target_yaw_rad,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for value in values
        ):
            return DoorMissionEvaluation(False, 0, False)

        position = sample.position_m
        velocity = sample.velocity_m_s
        target = sample.target_position_m
        planar_error = hypot(target[0] - position[0], target[1] - position[1])
        horizontal_speed = hypot(velocity[0], velocity[1])
        standoff = _door_standoff_m(
            position,
            sample.room_bounds_m,
            sample.door_face,
        )
        in_tolerance = bool(
            sample.visible
            and planar_error <= self.planar_position_tolerance_m
            and abs(target[2] - position[2])
            <= self.vertical_position_tolerance_m
            and abs(standoff - self.target_standoff_m)
            <= self.standoff_tolerance_m
            and abs(_wrap_angle(sample.yaw_rad - sample.target_yaw_rad))
            <= self.yaw_alignment_tolerance_rad
            and horizontal_speed <= self.max_horizontal_speed_m_s
            and abs(velocity[2]) <= self.max_vertical_speed_m_s
            and abs(sample.yaw_rate_rad_s) <= self.max_yaw_rate_rad_s
        )
        dwell = min(prior_dwell_steps + 1, self.dwell_steps) if in_tolerance else 0
        return DoorMissionEvaluation(
            in_tolerance,
            dwell,
            dwell >= self.dwell_steps,
        )


FIXED_DOOR_MISSION_METRIC_V1 = DoorMissionMetric(
    metric_id="fixed-door-approach-settle-0p80m-v1",
    schema_version=1,
    target_standoff_m=0.80,
    planar_position_tolerance_m=0.10,
    vertical_position_tolerance_m=0.10,
    standoff_tolerance_m=0.08,
    yaw_alignment_tolerance_rad=pi / 18.0,
    max_horizontal_speed_m_s=0.08,
    max_vertical_speed_m_s=0.05,
    max_yaw_rate_rad_s=pi / 36.0,
    dwell_steps=33,
)


def classify_mission_metric(
    report: Mapping[str, Any] | None,
) -> MissionMetricCompatibility:
    metric_id = report.get("metric_id") if isinstance(report, Mapping) else None
    if metric_id == FIXED_DOOR_MISSION_METRIC_V1.metric_id:
        return "current"
    return "incompatible"


def require_current_mission_metric(
    report: Mapping[str, Any] | None,
) -> DoorMissionMetric:
    compatibility = classify_mission_metric(report)
    if compatibility != "current":
        raise ValueError(
            "incompatible fixed-door mission metric cannot be used "
            "for promotion"
        )
    if dict(report or {}) != FIXED_DOOR_MISSION_METRIC_V1.to_report():
        raise ValueError("fixed-door mission metric fields or SHA-256 do not match")
    return FIXED_DOOR_MISSION_METRIC_V1


def _door_standoff_m(
    position_m: Sequence[float],
    room_bounds_m: Sequence[float],
    door_face: int,
) -> float:
    if door_face == 0:
        return position_m[0] - room_bounds_m[0]
    if door_face == 1:
        return room_bounds_m[1] - position_m[0]
    if door_face == 2:
        return position_m[1] - room_bounds_m[2]
    return room_bounds_m[3] - position_m[1]


def _wrap_angle(value: float) -> float:
    return (value + pi) % (2.0 * pi) - pi
