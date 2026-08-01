from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch

from flightrl.puffer4_door_observation import (
    DOOR_PHASE_DIM,
    DoorObservationOrigin,
    build_door_proprioception,
    door_observation_origin,
)
from flightrl.puffer4_door_evidence import detector_evidence
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_door_policy import (
    DOOR_HEIGHT,
    DOOR_OBS_DIM,
    DOOR_PRIVILEGED_DIM,
    DOOR_WIDTH,
)
from flightrl.puffer4_door_runtime_policy import DoorPufferRuntime
from flightrl.semantic.contract import GroundingDetection
from flightrl.puffer4_door_self_mask import apply_door_self_mask
from flightrl.puffer4_door_policy_contract import DoorPolicyArchitecture


class DoorFrameEncoder:
    """Host implementation of native current/delta/motion preprocessing."""

    def __init__(self) -> None:
        self.previous: np.ndarray | None = None

    def reset(self) -> None:
        self.previous = None

    def encode(self, frame: np.ndarray) -> np.ndarray:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("L")
        image = image.resize((DOOR_WIDTH, DOOR_HEIGHT), Image.Resampling.BILINEAR)
        current = np.asarray(image, dtype=np.float32)
        current = np.clip(
            17.0 * np.rint(current / 17.0),
            0.0,
            255.0,
        ).astype(np.uint8)
        current = apply_door_self_mask(current)
        delta = (
            np.zeros_like(current)
            if self.previous is None
            else (
                current.astype(np.float32)
                - self.previous.astype(np.float32)
            )
            / 255.0
        )
        motion = (np.abs(delta) >= 0.08).astype(np.float32)
        self.previous = current
        return np.concatenate(
            ((current / 255.0).reshape(-1), delta.reshape(-1), motion.reshape(-1))
        ).astype(np.float32)


class DoorPhase(IntEnum):
    SEARCH = 0
    TRACK = 1
    APPROACH = 2
    RECOVER = 3


@dataclass(frozen=True, slots=True)
class PhaseState:
    name: str
    one_hot: np.ndarray
    target_scale: float
    evidence: np.ndarray


class DoorMissionPhase:
    def __init__(
        self,
        *,
        approach_scale: float = 0.55,
    ) -> None:
        if not 0.0 < approach_scale <= 1.0:
            raise ValueError("approach_scale must be in (0, 1]")
        self.approach_scale = approach_scale
        self.maximum_detection_age_s = (
            FIXED_DOOR_EVIDENCE_AGE_CONTRACT.maximum_evidence_age_s
        )
        self.target_seen = False

    def reset(self) -> None:
        self.target_seen = False

    def update(
        self,
        detection: GroundingDetection | None,
        *,
        age_s: float | None = 0.0,
    ) -> PhaseState:
        evidence = detector_evidence(
            detection,
            age_s=age_s,
            maximum_age_s=self.maximum_detection_age_s,
        )
        detected = evidence[0] > 0.0 and evidence[4] < 1.0
        scale = float(evidence[3])
        if detected:
            self.target_seen = True
            phase = (
                DoorPhase.APPROACH
                if scale >= self.approach_scale
                else DoorPhase.TRACK
            )
        else:
            phase = DoorPhase.RECOVER if self.target_seen else DoorPhase.SEARCH
        one_hot = np.zeros(DOOR_PHASE_DIM, dtype=np.float32)
        one_hot[int(phase)] = 1.0
        return PhaseState(phase.name.lower(), one_hot, scale, evidence)


class DoorPufferShadow:
    """Runs a fixed-door checkpoint against camera and telemetry without control."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        architecture: DoorPolicyArchitecture | None = None,
    ) -> None:
        self.policy = DoorPufferRuntime.from_checkpoint(
            checkpoint,
            architecture,
        )
        self.frame = DoorFrameEncoder()
        self.phase = DoorMissionPhase()
        self.state = self.policy.initial_state()
        self.origin: DoorObservationOrigin | None = None
        self.previous_action = np.zeros(2, dtype=np.float32)

    @classmethod
    def from_state_dict(
        cls,
        state,
        *,
        architecture: DoorPolicyArchitecture | None = None,
    ) -> DoorPufferShadow:
        shadow = cls.__new__(cls)
        shadow.policy = DoorPufferRuntime.from_state_dict(
            state,
            architecture=architecture,
        )
        shadow.frame = DoorFrameEncoder()
        shadow.phase = DoorMissionPhase()
        shadow.state = shadow.policy.initial_state()
        shadow.origin = None
        shadow.previous_action = np.zeros(2, dtype=np.float32)
        return shadow

    def reset(self) -> None:
        self.frame.reset()
        self.phase.reset()
        self.state = self.policy.initial_state()
        self.origin = None
        self.previous_action.fill(0.0)

    @torch.no_grad()
    def step(
        self,
        frame: np.ndarray,
        telemetry: dict[str, float],
        *,
        detection: GroundingDetection | None,
        detection_age_s: float | None = 0.0,
        executed_previous_action: np.ndarray | None = None,
    ) -> dict[str, float | bool | str]:
        if self.origin is None:
            self.origin = door_observation_origin(telemetry)
        if executed_previous_action is not None:
            action = np.asarray(executed_previous_action, dtype=np.float32)
            if action.shape != (2,):
                raise ValueError("executed_previous_action must have shape (2,)")
            self.previous_action[:] = action
        phase = self.phase.update(detection, age_s=detection_age_s)
        proprio = build_door_proprioception(
            telemetry,
            self.origin,
            self.previous_action,
            phase.one_hot,
            phase.evidence,
        )
        observation = np.concatenate(
            (
                self.frame.encode(frame),
                proprio,
                np.zeros(DOOR_PRIVILEGED_DIM, dtype=np.float32),
            )
        )
        if observation.shape != (DOOR_OBS_DIM,):
            raise RuntimeError(f"door observation has invalid shape {observation.shape}")
        started = perf_counter()
        action, value, self.state = self.policy.forward_eval(
            torch.from_numpy(observation[None, :]),
            self.state,
        )
        inference_ms = 1_000.0 * (perf_counter() - started)
        bounded = action[0].numpy()
        bounded[0] = np.clip(bounded[0], 0.0, 1.0)
        bounded[1] = np.clip(bounded[1], -1.0, 1.0)
        return {
            "monitor_only": True,
            "controls_drone": False,
            "phase": phase.name,
            "target_detected": detection is not None,
            "target_scale": phase.target_scale,
            "action_forward": float(bounded[0]),
            "action_yaw": float(bounded[1]),
            "value": float(value[0, 0]),
            "inference_ms": inference_ms,
        }
