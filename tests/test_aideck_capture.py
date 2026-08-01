from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from flightrl.hardware.aideck_stream import AiDeckFrame
from flightrl.semantic.frame_integrity import FrameIntegrityError, load_frame_integrity_registry
from scripts import capture_aideck_vision as capture


class _FakeStream:
    dropped_frames = 0
    rejected_datagrams = 2

    def __init__(self, frames: tuple[AiDeckFrame, ...]) -> None:
        self._frames = frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass

    def frames(self, limit: int):
        yield from self._frames[:limit]


def test_udp_capture_is_durably_unreviewed_and_non_authoritative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "capture.npz"
    frame_dir = tmp_path / "frames"
    pixels = np.arange(6, dtype=np.uint8).reshape(2, 3)
    frames = (
        AiDeckFrame(1, 10.0, 3, 2, 1, 0, pixels),
        AiDeckFrame(2, 10.1, 3, 2, 1, 0, pixels),
    )
    monkeypatch.setattr(capture, "stream_from_args", lambda _args: _FakeStream(frames))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture_aideck_vision.py",
            "--transport",
            "udp",
            "--frames",
            "2",
            "--output",
            str(output),
            "--frame-dir",
            str(frame_dir),
        ],
    )

    capture.main()

    with np.load(output, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata_json"]))
        assert bool(artifact["complete"]) is True
        assert int(artifact["rejected_datagrams"]) == 2
    assert metadata["schema"] == "flightrl.aideck_decoded_frame_capture.v2"
    assert metadata["integrity_status"] == "unreviewed"
    assert metadata["training_authority"] is False
    assert metadata["deployment_authority"] is False
    assert metadata["transport_integrity"]["chunk_order_verified"] is False
    assert metadata["transport_integrity"]["firmware_sequence_field_present"] is False
    assert output.with_suffix(".npz.provenance.json").is_file()

    registry = load_frame_integrity_registry(frame_dir / "frame-integrity.json")
    with pytest.raises(FrameIntegrityError, match="unreviewed"):
        registry.require_frame_safe(frame_dir)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (AiDeckFrame(2, 1.0, 1, 1, 1, 0, np.zeros((1, 1), dtype=np.uint8)), "index"),
        (AiDeckFrame(1, float("nan"), 1, 1, 1, 0, np.zeros((1, 1), dtype=np.uint8)), "finite"),
        (AiDeckFrame(1, "1.0", 1, 1, 1, 0, np.zeros((1, 1), dtype=np.uint8)), "finite"),
        (AiDeckFrame(1, 1.0, 1, 1, 0, 0, np.zeros((1, 1), dtype=np.uint8)), "positive integer"),
        (AiDeckFrame(1, 1.0, 1, 1, 1, 9, np.zeros((1, 1), dtype=np.uint8)), "format"),
        (AiDeckFrame(1, 1.0, 1, 1, 1, 0, np.zeros((1, 1), dtype=np.float32)), "uint8"),
    ],
)
def test_capture_rejects_malformed_frame_metadata(frame: AiDeckFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        capture.validate_frame(frame, expected_index=1, previous_time_s=None)


def test_capture_rejects_decreasing_timestamp_and_invalid_counter() -> None:
    frame = AiDeckFrame(1, 0.9, 1, 1, 1, 0, np.zeros((1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="nondecreasing"):
        capture.validate_frame(frame, expected_index=1, previous_time_s=1.0)

    stream = type("BadStream", (), {"dropped_frames": float("nan")})()
    with pytest.raises(SystemExit, match="invalid dropped_frames"):
        capture.validated_counter(stream, "dropped_frames")
