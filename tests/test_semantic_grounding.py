from __future__ import annotations

import json
from time import time

import numpy as np
import pytest
from PIL import Image

from flightrl.hardware.aideck_stream import AiDeckFrame
from flightrl.semantic import (
    GroundingDetection,
    GroundingResult,
    NormalizedBox,
    ResolutionVariant,
    SemanticRunWriter,
    degrade_frame,
    require_semantic_frame,
)
from flightrl.semantic.clip_verifier import (
    ClipVerifierConfig,
    padded_crop,
    target_description,
)
from flightrl.semantic.grounding_dino import (
    GroundingDinoConfig,
    _candidate_rows,
    _detection,
    _matches_target,
)


def grounding_result(center_x: float = 0.5) -> GroundingResult:
    now = time()
    return GroundingResult(
        prompt="door",
        frame_index=4,
        frame_host_time_s=now,
        image_width=162,
        image_height=122,
        source_mean=72.0,
        inference_ms=20.0,
        detections=(
            GroundingDetection(
                "door",
                0.8,
                NormalizedBox(center_x - 0.1, 0.2, center_x + 0.1, 0.8),
            ),
        ),
    )


def test_semantic_writer_persists_non_actuating_evidence(tmp_path) -> None:
    result = grounding_result()
    frame = AiDeckFrame(
        index=result.frame_index,
        host_time_s=result.frame_host_time_s,
        width=162,
        height=122,
        depth=1,
        format=1,
        pixels=np.full((122, 162), 80, dtype=np.uint8),
    )

    with SemanticRunWriter(tmp_path, manifest={"target": "door"}) as writer:
        annotated = writer.write(
            frame,
            result,
            telemetry={"stateEstimate.z": 0.3},
        )

    event = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[0])
    assert annotated.exists()
    assert event["grounding"]["detections"][0]["label"] == "door"
    assert event["telemetry"]["stateEstimate.z"] == pytest.approx(0.3)
    assert event["controls_drone"] is False
    assert "command" not in event
    assert "policy_shadow" not in event


def test_grounding_result_distinguishes_proposals_from_verified_detections() -> None:
    proposal = GroundingDetection(
        "door",
        0.7,
        NormalizedBox(0.2, 0.2, 0.8, 0.8),
    )
    result = GroundingResult(
        prompt="door",
        frame_index=1,
        frame_host_time_s=time(),
        image_width=64,
        image_height=48,
        source_mean=50.0,
        inference_ms=10.0,
        detections=(),
        proposed_detections=(proposal,),
    )

    assert result.best is None
    assert result.best_proposal == proposal
    assert result.to_dict()["proposed_detections"][0]["label"] == "door"


@pytest.mark.parametrize(
    ("width", "pixels", "message"),
    (
        (64, np.full((48, 64), 80, dtype=np.uint8), "flash the semantic JPEG"),
        (162, np.zeros((122, 162), dtype=np.uint8), "too dark"),
    ),
)
def test_semantic_frame_gate_rejects_invalid_capture(
    width: int,
    pixels: np.ndarray,
    message: str,
) -> None:
    frame = AiDeckFrame(1, time(), width, pixels.shape[0], 1, 1, pixels)

    with pytest.raises(RuntimeError, match=message):
        require_semantic_frame(frame, min_width=128, min_mean=8.0)


def test_grounding_adapter_drops_invalid_detections() -> None:
    assert _detection("", 0.4, (1.0, 1.0, 5.0, 5.0), 10, 10) is None
    assert _detection("door", 0.4, (5.0, 1.0, 5.0, 5.0), 10, 10) is None
    assert _matches_target("door", "door")
    assert not _matches_target("door wall", "door")


def test_target_only_candidates_use_requested_target() -> None:
    processed = {
        "text_labels": ["monitor", "extra label"],
        "scores": [0.8],
        "boxes": [(1.0, 2.0, 3.0, 4.0)],
    }

    assert _candidate_rows(
        processed,
        target="computer monitor",
        target_only=True,
    ) == (("computer monitor", 0.8, (1.0, 2.0, 3.0, 4.0)),)


def test_resolution_degradation_has_requested_shape_and_bit_depth() -> None:
    source = np.arange(324 * 244, dtype=np.uint8).reshape(244, 324)

    gray4 = degrade_frame(source, ResolutionVariant(64, 48, 4))

    assert gray4.shape == (48, 64)
    assert len(np.unique(gray4)) <= 16


def test_clip_verifier_target_description_and_padding() -> None:
    config = ClipVerifierConfig()
    grounder_config = GroundingDinoConfig()
    detection = GroundingDetection(
        "computer monitor",
        0.8,
        NormalizedBox(0.8, 0.2, 1.0, 0.4),
    )
    image = Image.fromarray(np.zeros((100, 200), dtype=np.uint8))

    crop = padded_crop(image, detection, padding=0.5)

    assert target_description("computer monitor") == "a computer monitor on a desk"
    assert grounder_config.threshold == 0.25
    assert grounder_config.distractor_labels == ()
    assert config.minimum_probability == 0.60
    assert config.minimum_margin == 0.45
    assert crop.size == (60, 40)
