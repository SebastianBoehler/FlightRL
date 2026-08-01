from __future__ import annotations

from time import time

import numpy as np
import pytest

from flightrl.hardware.aideck_stream import AiDeckFrame
from flightrl.puffer4_door_shadow_io import REQUIRED_TELEMETRY
from flightrl.semantic import GroundingDetection, GroundingResult, NormalizedBox
from flightrl.semantic.controller import DiscoveryCommand, DiscoveryPhase
from flightrl.semantic.live import SEMANTIC_LOG_VARIABLES
from flightrl.semantic.readiness import (
    YAW_ONLY_AXIS_AUTHORITY,
    load_yaw_only_readiness,
)
from flightrl.semantic.yaw_authority import PufferYawAuthority


class _Policy:
    def __init__(self, yawrate: float) -> None:
        self.yawrate = yawrate
        self.executed = None
        self.step_arguments = None

    def step(self, **kwargs) -> dict:
        self.step_arguments = kwargs
        return {
            "controls_drone": False,
            "yawrate_deg_s": self.yawrate,
        }

    def record_executed_action(self, **action) -> None:
        self.executed = action


def test_puffer_authority_clamps_yaw_and_denies_translation() -> None:
    policy = _Policy(30.0)
    authority = PufferYawAuthority(policy, _readiness())
    now_s = time()
    proposal = authority.update(
        _frame(now_s),
        _grounding(now_s, detected=True),
        {},
        now_s=now_s,
    )
    command = authority.apply(_command(), now_s=now_s)

    assert proposal["applied_yawrate_deg_s"] == 8.0
    assert proposal["controls_drone"] is True
    assert command.vx_body_m_s == 0.0
    assert command.vy_body_m_s == 0.0
    assert command.yawrate_deg_s == 8.0
    assert policy.executed == {
        "vx_body_m_s": 0.0,
        "yawrate_deg_s": 8.0,
    }


def test_puffer_authority_stops_stale_or_terminal_proposal() -> None:
    authority = PufferYawAuthority(_Policy(15.0), _readiness())
    authority.update(_frame(10.0), _grounding(10.0), {}, now_s=10.0)

    assert authority.apply(_command(), now_s=11.1).yawrate_deg_s == 0.0
    assert (
        authority.apply(
            _command(DiscoveryPhase.TIMEOUT),
            now_s=10.1,
        ).yawrate_deg_s
        == 0.0
    )


def test_puffer_authority_forwards_grounding_age_to_policy() -> None:
    policy = _Policy(0.0)
    authority = PufferYawAuthority(policy, _readiness())

    authority.update(
        _frame(10.0),
        _grounding(8.5, detected=True),
        {},
        now_s=10.0,
    )

    assert policy.step_arguments["detection_age_s"] == pytest.approx(1.5)


def test_puffer_authority_does_not_refresh_stale_source_frame() -> None:
    authority = PufferYawAuthority(_Policy(7.0), _readiness())

    authority.update(
        _frame(8.5),
        _grounding(8.5),
        {},
        now_s=10.0,
    )

    assert authority.apply(_command(), now_s=10.0).yawrate_deg_s == 0.0


def test_semantic_flight_logging_covers_fixed_door_observation() -> None:
    assert set(REQUIRED_TELEMETRY).issubset(SEMANTIC_LOG_VARIABLES)


def test_readiness_rejects_checkpoint_hash_mismatch(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"actual")
    report = tmp_path / "readiness.json"
    report.write_text(
        __import__("json").dumps(
            {
                **_readiness(),
                "checkpoint_sha256": "incorrect",
                "next_live_gate_passed": True,
                "translation_authority_passed": False,
            }
        )
    )

    with pytest.raises(ValueError, match="SHA-256"):
        load_yaw_only_readiness(report, checkpoint)


def _readiness() -> dict:
    return {
        "limits": {
            "search_abs_yawrate_deg_s": 20.0,
            "detected_abs_yawrate_deg_s": 8.0,
            "proposal_stale_s": 1.0,
        },
        "axis_authority": YAW_ONLY_AXIS_AUTHORITY,
    }


def _frame(now_s: float) -> AiDeckFrame:
    return AiDeckFrame(
        index=1,
        host_time_s=now_s,
        width=64,
        height=48,
        depth=1,
        format=2,
        pixels=np.zeros((48, 64), dtype=np.uint8),
    )


def _grounding(now_s: float, *, detected: bool = False) -> GroundingResult:
    detections = ()
    if detected:
        detections = (
            GroundingDetection(
                "monitor",
                0.8,
                NormalizedBox(0.1, 0.1, 0.3, 0.8),
            ),
        )
    return GroundingResult(
        prompt="monitor",
        frame_index=1,
        frame_host_time_s=now_s,
        image_width=64,
        image_height=48,
        source_mean=60.0,
        inference_ms=10.0,
        detections=detections,
    )


def _command(phase: DiscoveryPhase = DiscoveryPhase.SCAN) -> DiscoveryCommand:
    return DiscoveryCommand(
        phase=phase,
        vx_body_m_s=0.2,
        vy_body_m_s=-0.2,
        yawrate_deg_s=-5.0,
        target_visible=False,
        target_confidence=0.0,
        horizontal_error=0.0,
        waypoint_index=0,
    )
