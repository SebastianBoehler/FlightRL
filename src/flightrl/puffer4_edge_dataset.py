from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np
import torch

from flightrl.puffer4_edge_contract import (
    edge_target_one_hot,
    validate_edge_target_id,
    validate_normalized_edge_action,
    validate_normalized_edge_telemetry,
)
from flightrl.puffer4_edge_schema import (
    EDGE_ACTION_DIM,
    EDGE_FRAME_PIXELS,
    EDGE_MISSION_TOKEN_COUNT,
    EDGE_OBSERVATION_DIM,
    EDGE_TELEMETRY_BOUNDS,
    EDGE_TELEMETRY_DIM,
)
from flightrl.puffer4_edge_wire_codec import pack_gray4, unpack_gray4


EDGE_STUDENT_TRAINING_TAIL_DIM = EDGE_ACTION_DIM + 4 + 1
EDGE_STUDENT_OBSERVATION_DIM = EDGE_OBSERVATION_DIM + EDGE_STUDENT_TRAINING_TAIL_DIM

pack_gray4_nibbles, unpack_gray4_nibbles = pack_gray4, unpack_gray4


@dataclass(frozen=True, slots=True)
class EdgeTeacherRecord:
    packed_frame: bytes
    telemetry: tuple[float, ...]
    target_id: int
    teacher_action: tuple[float, ...]
    grounding: tuple[float, ...]
    scene_group_id: int
    reset: bool
    done_after_action: bool

    def __post_init__(self) -> None:
        frame = bytes(self.packed_frame)
        unpack_gray4(frame)
        telemetry = validate_normalized_edge_telemetry(self.telemetry)
        action = validate_normalized_edge_action(self.teacher_action)
        grounding = _validate_grounding(self.grounding)
        object.__setattr__(self, "packed_frame", frame)
        object.__setattr__(self, "telemetry", telemetry)
        object.__setattr__(self, "teacher_action", action)
        object.__setattr__(self, "grounding", grounding)
        _validate_scene_group_id(self.scene_group_id)
        validate_edge_target_id(self.target_id)
        _validate_flag(self.reset, "reset")
        _validate_flag(self.done_after_action, "done-after-action")

    def model_observation(self) -> np.ndarray:
        frame = unpack_gray4(self.packed_frame).numpy()
        suffix = np.asarray(
            (*self.telemetry, *edge_target_one_hot(self.target_id)),
            dtype=np.float32,
        )
        observation = np.concatenate((frame, suffix))
        if observation.shape != (EDGE_OBSERVATION_DIM,):
            raise RuntimeError("edge teacher record violates the model ABI")
        return observation


@dataclass(frozen=True, slots=True)
class EdgeTeacherBatch:
    packed_frames: np.ndarray
    telemetry: np.ndarray
    target_ids: np.ndarray
    teacher_actions: np.ndarray
    grounding: np.ndarray
    scene_group_ids: np.ndarray


def adapt_native_door_observation(
    observation: np.ndarray | torch.Tensor,
    *,
    target_id: int,
    reset: bool,
    done_after_action: bool,
) -> EdgeTeacherRecord:
    _validate_flag(reset, "reset")
    _validate_flag(done_after_action, "done-after-action")
    validated_target = validate_edge_target_id(target_id)
    values = _native_values(observation)
    batch = adapt_native_door_observation_batch(values[None, :])
    if int(batch.target_ids[0]) != validated_target:
        raise ValueError("native edge target one-hot does not match target ID")
    return EdgeTeacherRecord(
        packed_frame=batch.packed_frames[0].tobytes(),
        telemetry=tuple(float(value) for value in batch.telemetry[0]),
        target_id=validated_target,
        teacher_action=tuple(float(value) for value in batch.teacher_actions[0]),
        grounding=tuple(float(value) for value in batch.grounding[0]),
        scene_group_id=int(batch.scene_group_ids[0]),
        reset=reset,
        done_after_action=done_after_action,
    )


def adapt_native_door_observation_batch(
    observations: np.ndarray | torch.Tensor,
) -> EdgeTeacherBatch:
    values = _native_batch_values(observations)
    frame_end = EDGE_FRAME_PIXELS
    telemetry_end = frame_end + EDGE_TELEMETRY_DIM
    actor_end = telemetry_end + EDGE_MISSION_TOKEN_COUNT
    action_end = actor_end + EDGE_ACTION_DIM
    grounding_end = action_end + 4
    frames = values[:, :frame_end]
    levels = frames * 15.0
    if np.any((frames < 0.0) | (frames > 1.0)) or not np.allclose(
        levels,
        np.rint(levels),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("native edge frame must contain exact gray4 levels")
    nibbles = np.rint(levels).astype(np.uint8)
    packed = (nibbles[:, 0::2] << 4) | nibbles[:, 1::2]
    telemetry = values[:, frame_end:telemetry_end]
    mission = values[:, telemetry_end:actor_end]
    actions = values[:, actor_end:action_end]
    grounding = values[:, action_end:grounding_end]
    scene_groups = values[:, grounding_end]
    _validate_batch_values(telemetry, mission, actions, grounding)
    if np.any(scene_groups != np.rint(scene_groups)) or np.any(
        (scene_groups < 0.0) | (scene_groups > 127.0)
    ):
        raise ValueError("native edge scene group ID is invalid")
    return EdgeTeacherBatch(
        packed_frames=packed.copy(),
        telemetry=telemetry.copy(),
        target_ids=np.argmax(mission, axis=1).astype(np.uint8),
        teacher_actions=actions.copy(),
        grounding=grounding.copy(),
        scene_group_ids=scene_groups.astype(np.uint8),
    )


def _native_values(observation: np.ndarray | torch.Tensor) -> np.ndarray:
    values = _native_array(observation)
    if values.shape != (EDGE_STUDENT_OBSERVATION_DIM,):
        raise ValueError(
            f"native edge observation must have shape ({EDGE_STUDENT_OBSERVATION_DIM},)"
        )
    return values


def _native_batch_values(observations: np.ndarray | torch.Tensor) -> np.ndarray:
    values = _native_array(observations)
    if (
        values.ndim != 2
        or not len(values)
        or values.shape[1] != EDGE_STUDENT_OBSERVATION_DIM
    ):
        raise ValueError(
            "native edge observations must have shape "
            f"[batch, {EDGE_STUDENT_OBSERVATION_DIM}]"
        )
    return values


def _native_array(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.requires_grad:
            raise ValueError("native edge observations must be detached on CPU")
        values = value.detach().numpy()
    else:
        values = np.asarray(value)
    if values.dtype != np.float32 or not bool(np.isfinite(values).all()):
        raise ValueError("native edge observations must be finite float32")
    return values


def _validate_batch_values(
    telemetry: np.ndarray,
    mission: np.ndarray,
    actions: np.ndarray,
    grounding: np.ndarray,
) -> None:
    bounds = np.asarray(EDGE_TELEMETRY_BOUNDS, dtype=np.float32)
    if np.any((telemetry < bounds[:, 0]) | (telemetry > bounds[:, 1])):
        raise ValueError("native edge telemetry is outside normalized bounds")
    for section, label in ((slice(6, 9), "body-up"), (slice(13, 15), "relative-yaw")):
        norm = np.linalg.norm(telemetry[:, section], axis=1)
        if not np.allclose(norm, 1.0, atol=1e-4, rtol=0.0):
            raise ValueError(f"native edge {label} vector is invalid")
    if np.any((mission != 0.0) & (mission != 1.0)) or np.any(
        mission.sum(axis=1) != 1.0
    ):
        raise ValueError("native edge mission token must be canonical one-hot")
    if np.any(np.abs(actions) > 1.0):
        raise ValueError("native edge teacher action is outside normalized bounds")
    visible = grounding[:, 0]
    if np.any((visible != 0.0) & (visible != 1.0)):
        raise ValueError("native edge grounding visibility must be binary")
    if np.any(np.abs(grounding[:, 1:3]) > 1.0) or np.any(
        (grounding[:, 3] < 0.0) | (grounding[:, 3] > 1.0)
    ):
        raise ValueError("native edge grounding is outside normalized bounds")
    if np.any(grounding[visible == 0.0, 1:] != 0.0):
        raise ValueError("absent native edge grounding must have zero box labels")
    if np.any(grounding[visible == 1.0, 3] <= 0.0):
        raise ValueError("visible native edge grounding must have positive scale")


def _validate_grounding(values: Sequence[object]) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("edge grounding has the wrong shape")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        for value in values
    ):
        raise ValueError("edge grounding must be finite numeric values")
    grounding = tuple(float(value) for value in values)
    visible, center_x, center_y, scale = grounding
    if visible not in (0.0, 1.0):
        raise ValueError("edge grounding visibility must be binary")
    if not -1.0 <= center_x <= 1.0 or not -1.0 <= center_y <= 1.0:
        raise ValueError("edge grounding center is outside [-1, 1]")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("edge grounding scale is outside [0, 1]")
    if visible == 0.0 and grounding[1:] != (0.0, 0.0, 0.0):
        raise ValueError("absent edge grounding must have zero box labels")
    if visible == 1.0 and scale <= 0.0:
        raise ValueError("visible edge grounding must have positive scale")
    return grounding


def _validate_flag(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"edge {label} flag must be boolean")
    return value


def _validate_scene_group_id(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 127:
        raise ValueError("edge scene group ID is invalid")
    return value
