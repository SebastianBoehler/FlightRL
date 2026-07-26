from __future__ import annotations

import json
from time import time

import numpy as np
import pytest

from flightrl.hardware.aideck_stream import AiDeckFrame
from flightrl.semantic.grounding_dino import _detection, _matches_target
from flightrl.semantic import (
    DiscoveryConfig,
    DiscoveryController,
    DiscoveryPhase,
    GroundingDetection,
    GroundingResult,
    NormalizedBox,
    SemanticRunWriter,
    require_semantic_frame,
)


def _result(
    center_x: float,
    *,
    confidence: float = 0.8,
    frame_time_s: float = 10.0,
) -> GroundingResult:
    box = NormalizedBox(center_x - 0.1, 0.2, center_x + 0.1, 0.8)
    return GroundingResult(
        prompt="door",
        frame_index=4,
        frame_host_time_s=frame_time_s,
        image_width=162,
        image_height=122,
        source_mean=72.0,
        inference_ms=20.0,
        detections=(GroundingDetection("door", confidence, box),),
    )


def test_discovery_scans_then_tracks_and_holds_target() -> None:
    config = DiscoveryConfig(centered_hold_s=1.0, max_duration_s=30.0)
    controller = DiscoveryController(config, start_time_s=10.0)

    scan = controller.step(
        now_s=10.1,
        grounding=None,
        position_xy_m=(0.0, 0.0),
        origin_xy_m=(0.0, 0.0),
        yaw_deg=0.0,
    )
    track = controller.step(
        now_s=10.2,
        grounding=_result(0.8, frame_time_s=10.2),
        position_xy_m=(0.0, 0.0),
        origin_xy_m=(0.0, 0.0),
        yaw_deg=0.0,
    )
    hold = controller.step(
        now_s=10.3,
        grounding=_result(0.5, frame_time_s=10.3),
        position_xy_m=(0.0, 0.0),
        origin_xy_m=(0.0, 0.0),
        yaw_deg=0.0,
    )
    complete = controller.step(
        now_s=11.4,
        grounding=_result(0.5, frame_time_s=11.4),
        position_xy_m=(0.0, 0.0),
        origin_xy_m=(0.0, 0.0),
        yaw_deg=0.0,
    )

    assert scan.phase is DiscoveryPhase.SCAN
    assert scan.yawrate_deg_s == config.search_yawrate_deg_s
    assert track.phase is DiscoveryPhase.TRACK
    assert track.yawrate_deg_s < 0.0
    assert hold.phase is DiscoveryPhase.HOLD
    assert complete.phase is DiscoveryPhase.COMPLETE


def test_discovery_repositions_only_when_explicitly_enabled() -> None:
    disabled = DiscoveryController(
        DiscoveryConfig(search_yawrate_deg_s=20.0, max_duration_s=30.0),
        start_time_s=0.0,
    )
    stopped = disabled.step(
        now_s=18.1,
        grounding=None,
        position_xy_m=(0.0, 0.0),
        origin_xy_m=(0.0, 0.0),
        yaw_deg=0.0,
    )
    enabled = DiscoveryController(
        DiscoveryConfig(
            search_yawrate_deg_s=20.0,
            max_duration_s=30.0,
            allow_reposition=True,
        ),
        start_time_s=0.0,
    )
    moving = enabled.step(
        now_s=18.1,
        grounding=None,
        position_xy_m=(0.0, 0.0),
        origin_xy_m=(0.0, 0.0),
        yaw_deg=0.0,
    )

    assert stopped.phase is DiscoveryPhase.TIMEOUT
    assert moving.phase is DiscoveryPhase.REPOSITION
    assert moving.vx_body_m_s > 0.0
    assert moving.vy_body_m_s == pytest.approx(0.0)


def test_stale_or_low_confidence_grounding_does_not_control_yaw() -> None:
    controller = DiscoveryController(DiscoveryConfig(), start_time_s=10.0)

    stale = controller.step(
        now_s=13.0,
        grounding=_result(0.8, frame_time_s=10.0),
        position_xy_m=(0.0, 0.0),
        origin_xy_m=(0.0, 0.0),
        yaw_deg=0.0,
    )

    assert stale.phase is DiscoveryPhase.SCAN
    assert not stale.target_visible


def test_discovery_ignores_early_detection_until_initial_scan_finishes() -> None:
    controller = DiscoveryController(
        DiscoveryConfig(minimum_scan_s=5.0, max_duration_s=20.0),
        start_time_s=10.0,
    )

    scanning = controller.step(
        now_s=12.0,
        grounding=_result(0.8, frame_time_s=12.0),
        position_xy_m=(0.0, 0.0),
        origin_xy_m=(0.0, 0.0),
        yaw_deg=0.0,
    )
    reacquiring = controller.step(
        now_s=15.1,
        grounding=_result(0.8, frame_time_s=15.1),
        position_xy_m=(0.0, 0.0),
        origin_xy_m=(0.0, 0.0),
        yaw_deg=90.0,
    )
    tracking = controller.step(
        now_s=15.2,
        grounding=_result(0.8, frame_time_s=15.2),
        position_xy_m=(0.0, 0.0),
        origin_xy_m=(0.0, 0.0),
        yaw_deg=0.0,
    )

    assert scanning.phase is DiscoveryPhase.SCAN
    assert scanning.yawrate_deg_s == 20.0
    assert not scanning.target_visible
    assert reacquiring.phase is DiscoveryPhase.REACQUIRE
    assert reacquiring.yawrate_deg_s < 0.0
    assert tracking.phase is DiscoveryPhase.TRACK


def test_semantic_writer_persists_frame_detection_and_control(tmp_path) -> None:
    result = _result(0.5, frame_time_s=time())
    frame = AiDeckFrame(
        index=result.frame_index,
        host_time_s=result.frame_host_time_s,
        width=162,
        height=122,
        depth=1,
        format=1,
        pixels=np.full((122, 162), 80, dtype=np.uint8),
    )
    controller = DiscoveryController(DiscoveryConfig(), start_time_s=result.frame_host_time_s)
    command = controller.step(
        now_s=result.frame_host_time_s,
        grounding=result,
        position_xy_m=(0.0, 0.0),
        origin_xy_m=(0.0, 0.0),
        yaw_deg=0.0,
    )

    with SemanticRunWriter(tmp_path, manifest={"target": "door"}) as writer:
        annotated = writer.write(
            frame,
            result,
            command=command,
            telemetry={"stateEstimate.z": 0.3},
            controls_drone=False,
        )

    event = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[0])
    assert annotated.exists()
    assert (tmp_path / "frames" / "frame-000004.png").exists()
    assert event["grounding"]["detections"][0]["label"] == "door"
    assert event["command"]["phase"] == "hold"
    assert event["controls_drone"] is False


def test_semantic_frame_gate_rejects_policy_resolution() -> None:
    frame = AiDeckFrame(
        index=1,
        host_time_s=time(),
        width=64,
        height=48,
        depth=1,
        format=2,
        pixels=np.full((48, 64), 80, dtype=np.uint8),
    )

    with pytest.raises(RuntimeError, match="flash the semantic JPEG profile"):
        require_semantic_frame(frame, min_width=128, min_mean=8.0)


def test_semantic_frame_gate_rejects_dark_capture() -> None:
    frame = AiDeckFrame(
        index=1,
        host_time_s=time(),
        width=162,
        height=122,
        depth=1,
        format=1,
        pixels=np.zeros((122, 162), dtype=np.uint8),
    )

    with pytest.raises(RuntimeError, match="too dark"):
        require_semantic_frame(frame, min_width=128, min_mean=8.0)


def test_grounding_adapter_drops_empty_labels_and_degenerate_boxes() -> None:
    assert _detection("", 0.4, (1.0, 1.0, 5.0, 5.0), 10, 10) is None
    assert _detection("door", 0.4, (5.0, 1.0, 5.0, 5.0), 10, 10) is None
    assert _matches_target("door", "door")
    assert not _matches_target("door wall", "door")
