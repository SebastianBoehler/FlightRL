from __future__ import annotations

from math import isfinite

import numpy as np

from flightrl.puffer4_edge_schema import EDGE_FRAME_PIXELS, EDGE_HEIGHT, EDGE_WIDTH
from flightrl.sixdof.geometry import quat_to_matrix
from flightrl.sixdof.orientation import quat_to_yaw

from .contract import COVERAGE_MAXIMUM_YAW_RATE_DEG_S, COVERAGE_OBSERVATION_DIM


def build_coverage_observation(
    frame_gray4: np.ndarray,
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    quaternion: np.ndarray,
    body_rates: np.ndarray,
    takeoff_origin_z: float,
    mission_origin_position: np.ndarray,
    mission_origin_yaw: np.ndarray,
    previous_edge_action: np.ndarray,
) -> np.ndarray:
    frames = np.asarray(frame_gray4)
    if frames.dtype != np.uint8 or frames.ndim != 3 or frames.shape[1:] != (
        EDGE_HEIGHT,
        EDGE_WIDTH,
    ):
        raise ValueError("coverage frames must have shape [batch, 48, 64] and dtype uint8")
    if np.any(frames % 17):
        raise ValueError("coverage frames must contain exact decoded gray4 levels")
    batch = frames.shape[0]
    positions = _float_batch("position", position, batch, 3)
    velocities = _float_batch("velocity", velocity, batch, 3)
    quaternions = _float_batch("quaternion", quaternion, batch, 4)
    rates = _float_batch("body_rates", body_rates, batch, 3)
    origins = _float_batch("mission_origin_position", mission_origin_position, batch, 3)
    origin_yaws = _float_vector("mission_origin_yaw", mission_origin_yaw, batch)
    previous = _float_batch("previous_edge_action", previous_edge_action, batch, 4)
    if np.any(np.abs(previous) > 1.0):
        raise ValueError("previous edge action must stay inside normalized bounds")
    if isinstance(takeoff_origin_z, bool) or not isfinite(float(takeoff_origin_z)):
        raise ValueError("takeoff origin Z must be finite")

    rotation = quat_to_matrix(quaternions)
    body_velocity = np.einsum("nji,nj->ni", rotation, velocities, optimize=True)
    telemetry = np.empty((batch, 19), dtype=np.float32)
    telemetry[:, :3] = np.clip(
        body_velocity / np.asarray((1.0, 1.0, 0.5), dtype=np.float32),
        -1.0,
        1.0,
    )
    telemetry[:, 3:6] = np.clip(
        rates / np.asarray((6.0, 6.0, 4.0), dtype=np.float32),
        -1.0,
        1.0,
    )
    telemetry[:, 6:9] = rotation[:, 2, :]
    telemetry[:, 9] = np.clip(
        (positions[:, 2] - float(takeoff_origin_z)) / 2.5,
        0.0,
        1.0,
    )
    delta = positions - origins
    cosine, sine = np.cos(origin_yaws), np.sin(origin_yaws)
    telemetry[:, 10] = np.clip(
        (cosine * delta[:, 0] + sine * delta[:, 1]) / 4.0,
        -1.0,
        1.0,
    )
    telemetry[:, 11] = np.clip(
        (-sine * delta[:, 0] + cosine * delta[:, 1]) / 4.0,
        -1.0,
        1.0,
    )
    telemetry[:, 12] = np.clip(delta[:, 2] / 2.0, -1.0, 1.0)
    relative_yaw = quat_to_yaw(quaternions) - origin_yaws
    telemetry[:, 13] = np.sin(relative_yaw)
    telemetry[:, 14] = np.cos(relative_yaw)
    telemetry[:, 15:] = previous

    observation = np.empty((batch, COVERAGE_OBSERVATION_DIM), dtype=np.float32)
    observation[:, :EDGE_FRAME_PIXELS] = frames.reshape(batch, -1) / 255.0
    observation[:, EDGE_FRAME_PIXELS:] = telemetry
    return observation


def coverage_action_to_edge_feedback(action: np.ndarray) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("coverage action must contain four finite values")
    if np.any(np.abs(values) > 1.0):
        raise ValueError("coverage action must stay inside normalized bounds")
    if values[1] != 0.0 or values[2] != 0.0:
        raise ValueError("coverage vy and vz actions must be structurally zero")
    feedback = values.copy()
    feedback[3] *= COVERAGE_MAXIMUM_YAW_RATE_DEG_S / 45.0
    return feedback


def _float_batch(name: str, value: np.ndarray, batch: int, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (batch, width) or not np.isfinite(array).all():
        raise ValueError(f"coverage {name} must have shape [{batch}, {width}] and be finite")
    return array


def _float_vector(name: str, value: np.ndarray, batch: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (batch,) or not np.isfinite(array).all():
        raise ValueError(f"coverage {name} must have shape [{batch}] and be finite")
    return array
