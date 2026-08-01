from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np

from flightrl.puffer4_door_evidence import DOOR_EVIDENCE_DIM


DOOR_SENSOR_DIM = 17
DOOR_PHASE_DIM = 4
DOOR_PROPRIO_DIM = DOOR_SENSOR_DIM + DOOR_PHASE_DIM + DOOR_EVIDENCE_DIM


@dataclass(frozen=True, slots=True)
class DoorObservationOrigin:
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float


def door_observation_origin(telemetry: dict[str, float]) -> DoorObservationOrigin:
    return DoorObservationOrigin(
        x_m=_required(telemetry, "stateEstimate.x"),
        y_m=_required(telemetry, "stateEstimate.y"),
        z_m=_required(telemetry, "stateEstimate.z"),
        yaw_rad=radians(
            _required_any(telemetry, "stateEstimate.yaw", "stabilizer.yaw")
        ),
    )


def build_door_proprioception(
    telemetry: dict[str, float],
    origin: DoorObservationOrigin,
    previous_action: np.ndarray,
    phase: np.ndarray,
    evidence: np.ndarray,
    *,
    room_height_m: float = 2.5,
) -> np.ndarray:
    if room_height_m <= 0.0:
        raise ValueError("room_height_m must be positive")
    action = np.asarray(previous_action, dtype=np.float32)
    if action.shape != (2,):
        raise ValueError("previous_action must have shape (2,)")
    phase_values = np.asarray(phase, dtype=np.float32)
    if phase_values.shape != (DOOR_PHASE_DIM,):
        raise ValueError(f"phase must have shape ({DOOR_PHASE_DIM},)")
    evidence_values = np.asarray(evidence, dtype=np.float32)
    if evidence_values.shape != (DOOR_EVIDENCE_DIM,):
        raise ValueError(f"evidence must have shape ({DOOR_EVIDENCE_DIM},)")
    roll = radians(
        _required_any(telemetry, "stateEstimate.roll", "stabilizer.roll")
    )
    pitch = radians(
        _required_any(telemetry, "stateEstimate.pitch", "stabilizer.pitch")
    )
    yaw = radians(
        _required_any(telemetry, "stateEstimate.yaw", "stabilizer.yaw")
    )
    rotation = _euler_matrix(roll, pitch, yaw)
    velocity = np.asarray(
        (
            _required(telemetry, "stateEstimate.vx"),
            _required(telemetry, "stateEstimate.vy"),
            _required(telemetry, "stateEstimate.vz"),
        ),
        dtype=np.float32,
    )
    body_velocity = rotation.T @ velocity
    position = np.asarray(
        (
            _required(telemetry, "stateEstimate.x"),
            _required(telemetry, "stateEstimate.y"),
            _required(telemetry, "stateEstimate.z"),
        ),
        dtype=np.float32,
    )
    displacement = position - np.asarray(
        (origin.x_m, origin.y_m, origin.z_m),
        dtype=np.float32,
    )
    origin_cosine = cos(origin.yaw_rad)
    origin_sine = sin(origin.yaw_rad)
    local_displacement = np.asarray(
        (
            origin_cosine * displacement[0] + origin_sine * displacement[1],
            -origin_sine * displacement[0] + origin_cosine * displacement[1],
            displacement[2],
        ),
        dtype=np.float32,
    )
    relative_yaw = yaw - origin.yaw_rad
    rates = np.asarray(
        tuple(
            radians(_required(telemetry, f"gyro.{axis}")) / maximum
            for axis, maximum in zip(
                ("x", "y", "z"),
                (6.0, 6.0, 4.0),
                strict=True,
            )
        ),
        dtype=np.float32,
    )
    return np.asarray(
        (
            *np.clip(
                body_velocity / np.asarray((1.0, 1.0, 0.5)),
                -1.0,
                1.0,
            ),
            *np.clip(rates, -1.0, 1.0),
            *rotation[2],
            np.clip(position[2] / room_height_m, 0.0, 1.0),
            *np.clip(
                local_displacement / np.asarray((4.0, 4.0, 2.0)),
                -1.0,
                1.0,
            ),
            sin(relative_yaw),
            cos(relative_yaw),
            *action,
            *phase_values,
            *evidence_values,
        ),
        dtype=np.float32,
    )


def _euler_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float32,
    )


def _required(telemetry: dict[str, float], key: str) -> float:
    if key not in telemetry:
        raise KeyError(f"missing required telemetry variable: {key}")
    return float(telemetry[key])


def _required_any(telemetry: dict[str, float], *keys: str) -> float:
    for key in keys:
        if key in telemetry:
            return float(telemetry[key])
    raise KeyError(f"missing required telemetry variable: {' or '.join(keys)}")
