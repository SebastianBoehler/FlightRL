from __future__ import annotations

import hashlib
import json
from time import time

import numpy as np
import pytest

from flightrl.hardware.aideck_stream import AiDeckFrame
from flightrl.semantic import GroundingResult
from flightrl.semantic.bounded_authority import PufferBoundedAuthority
from flightrl.semantic.controller import DiscoveryCommand, DiscoveryPhase
from flightrl.semantic.readiness import (
    BOUNDED_FORWARD_AXIS_AUTHORITY,
    build_bounded_forward_readiness,
    load_bounded_forward_readiness,
    write_readiness,
)


class _Policy:
    def __init__(
        self,
        *,
        forward: float = 0.2,
        yawrate: float = 30.0,
        clearance: float = 1.0,
        collision_risk: float = 0.1,
    ) -> None:
        self.result = {
            "controls_drone": False,
            "vx_body_m_s": forward,
            "yawrate_deg_s": yawrate,
            "predicted_clearance_m": clearance,
            "predicted_collision_risk": collision_risk,
        }

    def step(self, **_kwargs) -> dict:
        return self.result.copy()


def test_bounded_authority_clamps_axes_and_stops_on_visual_risk() -> None:
    now_s = time()
    authority = PufferBoundedAuthority(_Policy(), _readiness())
    proposal = authority.update(
        _frame(now_s),
        _grounding(now_s),
        _telemetry(),
        now_s=now_s,
    )
    command = authority.apply(_command(), now_s=now_s)

    assert proposal["approved_authority"] == "bounded_forward_yaw"
    assert proposal["applied_vx_body_m_s"] == 0.05
    assert proposal["applied_vy_body_m_s"] == 0.0
    assert proposal["applied_vz_m_s"] == 0.0
    assert proposal["applied_yawrate_deg_s"] == 15.0
    assert command.vx_body_m_s == 0.05
    assert command.vy_body_m_s == 0.0

    unsafe = PufferBoundedAuthority(
        _Policy(clearance=0.2, collision_risk=0.8),
        _readiness(),
    )
    unsafe.update(_frame(now_s), _grounding(now_s), _telemetry(), now_s=now_s)
    assert unsafe.apply(_command(), now_s=now_s).vx_body_m_s == 0.0


def test_bounded_authority_stops_when_stale_or_budget_exhausted() -> None:
    authority = PufferBoundedAuthority(_Policy(), _readiness())
    authority.update(_frame(10.0), _grounding(10.0), _telemetry(), now_s=10.0)

    assert authority.apply(_command(), now_s=10.6).vx_body_m_s == 0.0
    proposal = authority.update(
        _frame(10.7),
        _grounding(10.7),
        _telemetry(x=0.21),
        now_s=10.7,
    )
    assert proposal["authority_budget_exhausted"] is True
    assert proposal["applied_vx_body_m_s"] == 0.0


def test_bounded_readiness_rejects_failed_sim_gate(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    training = tmp_path / "training.json"
    replay = tmp_path / "replay.json"
    training.write_text(json.dumps(_training_report(checkpoint, gate=False)))
    replay.write_text(json.dumps({"translation_shadow_gate_passed": True}))
    report = build_bounded_forward_readiness(checkpoint, training, replay)
    report_path = write_readiness(tmp_path / "readiness.json", report)

    assert report["sim_translation_gate_passed"] is False
    assert report["translation_authority_passed"] is False
    with pytest.raises(ValueError, match="translation authority gate"):
        load_bounded_forward_readiness(report_path, checkpoint)


def test_bounded_readiness_binds_checkpoint_and_evidence(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    training = tmp_path / "training.json"
    replay = tmp_path / "replay.json"
    training.write_text(json.dumps(_training_report(checkpoint, gate=True)))
    replay.write_text(json.dumps({"translation_shadow_gate_passed": True}))
    report = build_bounded_forward_readiness(checkpoint, training, replay)
    report_path = write_readiness(tmp_path / "readiness.json", report)

    loaded = load_bounded_forward_readiness(report_path, checkpoint)

    assert loaded["translation_authority_passed"] is True
    assert loaded["axis_authority"] == BOUNDED_FORWARD_AXIS_AUTHORITY


def _readiness() -> dict:
    return {
        "limits": {
            "max_forward_speed_m_s": 0.05,
            "search_abs_yawrate_deg_s": 15.0,
            "detected_abs_yawrate_deg_s": 8.0,
            "proposal_stale_s": 0.5,
            "max_authority_duration_s": 3.0,
            "max_displacement_m": 0.20,
            "minimum_predicted_clearance_m": 0.45,
            "maximum_predicted_collision_risk": 0.35,
        }
    }


def _training_report(checkpoint, *, gate: bool) -> dict:
    return {
        "active_exploration": True,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "selected_stage": "puffer_ppo",
        "shadow_gate_passed": gate,
        "evaluation": {
            "puffer_ppo": {
                "full": {
                    "success_rate": 0.6,
                    "target_discovery_rate": 0.8,
                    "collision_rate": 0.0,
                    "unsafe_forward_fraction": 0.01,
                    "minimum_moving_front_clearance_m": 0.3,
                    "clearance_false_safe_fraction": 0.01,
                    "max_lateral_vertical_action": 0.0,
                }
            }
        },
    }


def _frame(now_s: float) -> AiDeckFrame:
    return AiDeckFrame(1, now_s, 128, 96, 1, 2, np.zeros((96, 128), dtype=np.uint8))


def _grounding(now_s: float) -> GroundingResult:
    return GroundingResult("monitor", 1, now_s, 128, 96, 60.0, 10.0, ())


def _telemetry(*, x: float = 0.0) -> dict[str, float]:
    return {"stateEstimate.x": x, "stateEstimate.y": 0.0}


def _command() -> DiscoveryCommand:
    return DiscoveryCommand(
        DiscoveryPhase.SCAN,
        0.0,
        0.0,
        0.0,
        False,
        0.0,
        0.0,
        0,
    )
