from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import gymnasium
import numpy as np
import pytest
import torch

from flightrl.mujoco.semantic_observation import SemanticStudentObservationLayout
from flightrl.mujoco.semantic_vision_policy import SemanticVisionPolicy
from flightrl.navigation.spatial_memory import SpatialMemoryConfig
from flightrl.semantic.puffer_shadow import (
    SemanticPufferShadow,
    SemanticShadowConfig,
)
from flightrl.semantic.shadow_gate import (
    semantic_shadow_gate,
    semantic_translation_shadow_gate,
)
from flightrl.vision import VisionObservationConfig


def test_semantic_puffer_shadow_never_controls_drone(tmp_path) -> None:
    vision = VisionObservationConfig(
        width=64,
        height=48,
        color_mode="grayscale",
        include_delta=True,
        include_motion_mask=True,
        normalization="minus_one_one",
    )
    memory = SpatialMemoryConfig(cell_size_m=0.5, local_size=16)
    layout = SemanticStudentObservationLayout(vision, memory)
    contract = SimpleNamespace(
        single_observation_space=gymnasium.spaces.Box(
            -1.0,
            1.0,
            shape=(layout.flat_dim,),
            dtype=np.float32,
        ),
        single_action_space=gymnasium.spaces.Box(
            -1.0,
            1.0,
            shape=(4,),
            dtype=np.float32,
        ),
        vision_config=vision,
        memory_config=memory,
    )
    checkpoint = tmp_path / "semantic.pt"
    torch.save(SemanticVisionPolicy(contract, hidden_size=32).state_dict(), checkpoint)
    shadow = SemanticPufferShadow(checkpoint)

    result = shadow.step(
        frame=np.full((122, 162), 80, dtype=np.uint8),
        telemetry={
            "stateEstimate.x": 0.0,
            "stateEstimate.y": 0.0,
            "stateEstimate.z": 0.8,
            "stateEstimate.vx": 0.0,
            "stateEstimate.vy": 0.0,
            "stateEstimate.vz": 0.0,
            "stateEstimate.yaw": 0.0,
            "gyro.x": 0.0,
            "gyro.y": 0.0,
            "gyro.z": 0.0,
        },
        prompt="computer monitor",
        detection={
            "confidence": 0.8,
            "box": {
                "x_min": 0.4,
                "x_max": 0.6,
                "y_min": 0.2,
                "y_max": 0.8,
            },
        },
    )

    assert result["monitor_only"] is True
    assert result["controls_drone"] is False
    assert result["target_detected"] is True
    assert result["target_category"] == "monitor"
    assert abs(float(result["vx_body_m_s"])) <= shadow.config.max_horizontal_speed_m_s
    assert abs(float(result["yawrate_deg_s"])) <= shadow.config.max_yawrate_deg_s


def test_semantic_shadow_loads_checkpoint_contract_from_training_report(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "semantic.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "active_exploration": True,
        "max_horizontal_speed_m_s": 0.15,
        "observation_contract": {
            "vision": {
                "width": 128,
                "height": 96,
            }
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    config = SemanticShadowConfig.from_training_report(report)

    assert config.vision_width == 128
    assert config.vision_height == 96
    assert config.max_horizontal_speed_m_s == 0.15
    assert config.semantic_action_mode == "active_exploration"
    mismatched = tmp_path / "mismatched.pt"
    mismatched.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="does not match"):
        SemanticPufferShadow.from_training_report(
            mismatched,
            report_path,
        )


def test_semantic_shadow_loads_legacy_shared_safety_heads(tmp_path) -> None:
    vision = VisionObservationConfig(
        width=128,
        height=96,
        color_mode="grayscale",
        include_delta=True,
        include_motion_mask=True,
        normalization="minus_one_one",
    )
    memory = SpatialMemoryConfig(cell_size_m=0.5, local_size=16)
    layout = SemanticStudentObservationLayout(vision, memory)
    contract = SimpleNamespace(
        single_observation_space=gymnasium.spaces.Box(
            -1.0,
            1.0,
            shape=(layout.flat_dim,),
            dtype=np.float32,
        ),
        single_action_space=gymnasium.spaces.Box(
            -1.0,
            1.0,
            shape=(4,),
            dtype=np.float32,
        ),
        vision_config=vision,
        memory_config=memory,
        semantic_action_mode="active_exploration",
    )
    checkpoint = tmp_path / "semantic.pt"
    torch.save(
        SemanticVisionPolicy(
            contract,
            hidden_size=32,
            shared_visual_safety=True,
        ).state_dict(),
        checkpoint,
    )
    report = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "active_exploration": True,
        "max_horizontal_speed_m_s": 0.15,
        "observation_contract": {"vision": {"width": 128, "height": 96}},
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    shadow = SemanticPufferShadow.from_training_report(checkpoint, report_path)

    assert shadow.vision_config.shape == (3, 96, 128)
    assert shadow.policy.clearance_head is not None
    assert shadow.policy.collision_risk_head is not None
    assert shadow.policy.visual_safety is None
    result = shadow.step(
        frame=np.full((244, 324), 80, dtype=np.uint8),
        telemetry={},
        prompt="monitor",
        detection=None,
    )
    assert np.isfinite(result["predicted_clearance_m"])
    assert 0.0 <= result["predicted_collision_risk"] <= 1.0


def test_semantic_shadow_gate_requires_safe_no_detection_behavior() -> None:
    rows = [
        _gate_row(
            target_detected=True,
            target_acquired=True,
            horizontal_error=-0.3,
            vx=0.05,
            yawrate=8.0,
        )
        for _ in range(5)
    ]
    suppressed = [_gate_row() for _ in range(5)]

    passed = semantic_shadow_gate(rows, suppressed)
    unsafe = semantic_shadow_gate(
        rows,
        [_gate_row(vx=0.05) for _ in range(5)],
    )

    assert passed["next_live_shadow_gate_passed"] is True
    assert unsafe["next_live_shadow_gate_passed"] is False
    assert unsafe["suppressed_detection_horizontal_p95_m_s"] == 0.05


def test_translation_shadow_gate_requires_safety_predictions() -> None:
    rows = [
        {
            **_gate_row(target_detected=True, vx=0.05),
            "predicted_clearance_m": 1.0,
            "predicted_collision_risk": 0.1,
        }
        for _ in range(20)
    ]

    passed = semantic_translation_shadow_gate(rows)
    unsafe = semantic_translation_shadow_gate(
        [
            {
                **row,
                "predicted_clearance_m": 0.2,
                "predicted_collision_risk": 0.8,
            }
            for row in rows
        ]
    )

    assert passed["translation_shadow_gate_passed"] is True
    assert unsafe["translation_shadow_gate_passed"] is False
    assert unsafe["translation_unsafe_forward_fraction"] == 1.0


def _gate_row(
    *,
    target_detected: bool = False,
    target_acquired: bool = False,
    horizontal_error: float = 0.0,
    vx: float = 0.0,
    yawrate: float = 0.0,
) -> dict:
    return {
        "controls_drone": False,
        "target_detected": target_detected,
        "target_acquired": target_acquired,
        "detection_horizontal_error": horizontal_error,
        "vx_body_m_s": vx,
        "vy_body_m_s": 0.0,
        "vz_m_s": 0.0,
        "yawrate_deg_s": yawrate,
    }
