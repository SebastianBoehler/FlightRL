from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from math import radians
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium
import numpy as np
from PIL import Image
import torch

from flightrl.mujoco.semantic_observation import SemanticStudentObservationLayout
from flightrl.mujoco.semantic_vision_policy import SemanticVisionPolicy
from flightrl.navigation.room_generation import SEMANTIC_TARGET_CATEGORIES
from flightrl.navigation.semantic_scene import Bounds3D
from flightrl.navigation.spatial_memory import (
    EgocentricSpatialMemory,
    SpatialMemoryConfig,
)
from flightrl.vision import VisionObservationConfig, VisionObservationEncoder


@dataclass(frozen=True, slots=True)
class SemanticShadowConfig:
    vision_width: int = 64
    vision_height: int = 48
    assumed_target_distance_m: float = 2.0
    camera_horizontal_fov_deg: float = 82.0
    room_half_extent_m: float = 6.0
    room_height_m: float = 3.0
    max_horizontal_speed_m_s: float = 0.10
    max_vertical_speed_m_s: float = 0.10
    max_yawrate_deg_s: float = 60.0
    semantic_action_mode: str = "target_gated"

    def __post_init__(self) -> None:
        if self.vision_width <= 0 or self.vision_height <= 0:
            raise ValueError("semantic shadow vision dimensions must be positive")
        if self.semantic_action_mode not in {"target_gated", "active_exploration"}:
            raise ValueError("unknown semantic shadow action mode")

    @classmethod
    def from_training_report(cls, report: dict[str, Any]) -> SemanticShadowConfig:
        vision = report["observation_contract"]["vision"]
        return cls(
            vision_width=int(vision["width"]),
            vision_height=int(vision["height"]),
            max_horizontal_speed_m_s=float(report["max_horizontal_speed_m_s"]),
            semantic_action_mode=(
                "active_exploration"
                if report.get("active_exploration")
                else "target_gated"
            ),
        )


class SemanticPufferShadow:
    """Reconstruct semantic policy observations without issuing commands."""

    def __init__(
        self,
        checkpoint: str | Path,
        config: SemanticShadowConfig | None = None,
    ) -> None:
        self.config = config or SemanticShadowConfig()
        self.vision_config = VisionObservationConfig(
            width=self.config.vision_width,
            height=self.config.vision_height,
            color_mode="grayscale",
            include_delta=True,
            include_motion_mask=True,
            normalization="minus_one_one",
        )
        self.memory_config = SpatialMemoryConfig(cell_size_m=0.5, local_size=16)
        self.layout = SemanticStudentObservationLayout(
            self.vision_config,
            self.memory_config,
        )
        contract = SimpleNamespace(
            single_observation_space=gymnasium.spaces.Box(
                -1.0,
                1.0,
                shape=(self.layout.flat_dim,),
                dtype=np.float32,
            ),
            single_action_space=gymnasium.spaces.Box(
                -1.0,
                1.0,
                shape=(4,),
                dtype=np.float32,
            ),
            vision_config=self.vision_config,
            memory_config=self.memory_config,
            semantic_action_mode=self.config.semantic_action_mode,
        )
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        hidden_size = int(state_dict["encoder.fusion.0.weight"].shape[0])
        self.policy = SemanticVisionPolicy(
            contract,
            hidden_size=hidden_size,
            shared_visual_safety="clearance_head.weight" in state_dict,
            recurrent_safety=(
                "recurrent_safety.clearance_head.weight" in state_dict
            ),
            recurrent_visual_safety=(
                "recurrent_visual_safety.clearance_head.weight" in state_dict
            ),
        )
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        self.state = self.policy.initial_state(1, "cpu")
        self.vision = VisionObservationEncoder(self.vision_config)
        extent = self.config.room_half_extent_m
        self.memory = EgocentricSpatialMemory(
            Bounds3D(
                (-extent, -extent, 0.0),
                (extent, extent, self.config.room_height_m),
            ),
            self.memory_config,
        )

    @classmethod
    def from_training_report(
        cls,
        checkpoint: str | Path,
        report_path: str | Path,
        *,
        assumed_target_distance_m: float | None = None,
    ) -> SemanticPufferShadow:
        report = json.loads(Path(report_path).read_text())
        expected = report.get("checkpoint_sha256")
        if expected != _file_sha256(checkpoint):
            raise ValueError("semantic checkpoint does not match its training report")
        config = SemanticShadowConfig.from_training_report(report)
        if assumed_target_distance_m is not None:
            config = replace(
                config,
                assumed_target_distance_m=assumed_target_distance_m,
            )
        return cls(
            checkpoint,
            config,
        )

    @torch.no_grad()
    def step(
        self,
        *,
        frame: np.ndarray,
        telemetry: dict,
        prompt: str,
        detection: dict | None,
        update_semantic_memory: bool = True,
    ) -> dict[str, float | bool | str | None]:
        position = np.asarray(
            (
                _value(telemetry, "stateEstimate.x"),
                _value(telemetry, "stateEstimate.y"),
            ),
            dtype=np.float32,
        )
        yaw = radians(_value(telemetry, "stateEstimate.yaw", "stabilizer.yaw"))
        self.memory.update_pose(position)
        grounding_confidence = 0.0
        grounding_horizontal_error = 0.0
        if detection is not None:
            box = detection["box"]
            center_x = 0.5 * (float(box["x_min"]) + float(box["x_max"]))
            grounding_confidence = float(detection["confidence"])
            grounding_horizontal_error = center_x - 0.5
            bearing = (
                -radians(self.config.camera_horizontal_fov_deg)
                * grounding_horizontal_error
            )
            if update_semantic_memory:
                self.memory.observe_semantic(
                    position,
                    yaw,
                    bearing,
                    self.config.assumed_target_distance_m,
                    float(detection["confidence"]),
                    replace=True,
                )
        observation = np.empty(self.layout.flat_dim, dtype=np.float32)
        resized = np.asarray(
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize(
                (self.vision_config.width, self.vision_config.height),
                Image.Resampling.BILINEAR,
            )
        )
        observation[self.layout.vision_slice] = self.vision.encode_flat(resized)
        local_map = self.memory.local_map(
            position,
            yaw,
        )
        observation[self.layout.map_slice] = local_map.reshape(-1)
        observation[self.layout.proprioception_slice] = _proprioception(
            telemetry,
            yaw,
            self.config.room_height_m,
            grounding_confidence,
            grounding_horizontal_error,
        )
        observation[self.layout.command_slice] = _command_token(prompt)
        distribution, _, self.state, clearance_m, collision_risk = (
            self.policy.forward_eval_with_aux(
                torch.from_numpy(observation[None, :]),
                self.state,
            )
        )
        action = distribution.mean[0].clamp(-1.0, 1.0).numpy()
        return {
            "monitor_only": True,
            "controls_drone": False,
            "target_category": _target_category(prompt),
            "target_detected": detection is not None,
            "target_acquired": bool(
                local_map[self.policy.encoder.target_channel].max() > 0.0
            ),
            "action_vx": float(action[0]),
            "action_vy": float(action[1]),
            "action_vz": float(action[2]),
            "action_yaw": float(action[3]),
            "vx_body_m_s": float(action[0] * self.config.max_horizontal_speed_m_s),
            "vy_body_m_s": float(action[1] * self.config.max_horizontal_speed_m_s),
            "vz_m_s": float(action[2] * self.config.max_vertical_speed_m_s),
            "yawrate_deg_s": float(action[3] * self.config.max_yawrate_deg_s),
            "predicted_clearance_m": _optional_scalar(clearance_m),
            "predicted_collision_risk": _optional_scalar(collision_risk),
        }


def _proprioception(
    telemetry: dict,
    yaw: float,
    room_height_m: float,
    grounding_confidence: float,
    grounding_horizontal_error: float,
) -> np.ndarray:
    vx = _value(telemetry, "stateEstimate.vx")
    vy = _value(telemetry, "stateEstimate.vy")
    cosine, sine = np.cos(yaw), np.sin(yaw)
    body_velocity = (
        cosine * vx + sine * vy,
        -sine * vx + cosine * vy,
        _value(telemetry, "stateEstimate.vz"),
    )
    rates = tuple(
        radians(_value(telemetry, f"gyro.{axis}")) / maximum
        for axis, maximum in zip(("x", "y", "z"), (6.0, 6.0, 4.0), strict=True)
    )
    return np.asarray(
        (
            *np.clip(np.asarray(body_velocity) / 2.0, -1.0, 1.0),
            *np.clip(rates, -1.0, 1.0),
            np.clip(_value(telemetry, "stateEstimate.z") / room_height_m, 0.0, 1.0),
            np.sin(yaw),
            np.cos(yaw),
            0.0,
            0.0,
            grounding_confidence,
            grounding_horizontal_error,
        ),
        dtype=np.float32,
    )


def _target_category(prompt: str) -> str:
    normalized = prompt.strip().lower()
    aliases = {
        "door": ("door", "doorway"),
        "monitor": ("monitor", "display", "screen"),
        "sink": ("sink", "washbasin"),
    }
    for category, names in aliases.items():
        if any(name in normalized for name in names):
            return category
    raise ValueError(f"prompt must resolve to one of {SEMANTIC_TARGET_CATEGORIES}")


def _command_token(prompt: str) -> np.ndarray:
    token = np.zeros(len(SEMANTIC_TARGET_CATEGORIES), dtype=np.float32)
    token[SEMANTIC_TARGET_CATEGORIES.index(_target_category(prompt))] = 1.0
    return token


def _value(telemetry: dict, key: str, fallback: str | None = None) -> float:
    if key in telemetry:
        return float(telemetry[key])
    return float(telemetry.get(fallback, 0.0)) if fallback is not None else 0.0


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_scalar(value: torch.Tensor | None) -> float | None:
    return None if value is None else float(value.reshape(-1)[0])
