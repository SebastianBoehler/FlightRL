from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flightrl.semantic.door_real_evidence import load_real_door_evidence
from flightrl.semantic.frame_integrity import (
    FrameIntegrityError,
    load_frame_integrity_registry,
)


def _write_integrity_registry(root: Path) -> Path:
    path = root / "integrity.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "datasets": [
                    {
                        "path": "safe",
                        "status": "frame_safe",
                        "evidence": "reviewed",
                    },
                    {
                        "path": "corrupt",
                        "status": "known_corrupt",
                        "evidence": "mosaic",
                    },
                ],
            }
        )
    )
    return path


def test_real_evidence_loader_quantizes_frames_and_builds_box_labels(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe"
    safe.mkdir()
    Image.fromarray(np.full((96, 128), 101, dtype=np.uint8)).save(
        safe / "positive.png"
    )
    Image.fromarray(np.full((96, 128), 33, dtype=np.uint8)).save(
        safe / "negative.png"
    )
    manifest = tmp_path / "door-labels.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "samples": [
                    {
                        "frame": "safe/positive.png",
                        "visible": True,
                        "box": [0.25, 0.20, 0.75, 0.80],
                    },
                    {
                        "frame": "safe/negative.png",
                        "visible": False,
                    },
                ],
            }
        )
    )
    registry = load_frame_integrity_registry(
        _write_integrity_registry(tmp_path),
        root=tmp_path,
    )

    evidence = load_real_door_evidence(
        manifest,
        root=tmp_path,
        integrity_registry=registry,
    )

    assert evidence.frames.shape == (2, 1, 48, 64)
    assert np.all(np.isin(evidence.frames * 15.0, np.arange(16)))
    assert evidence.labels[0].tolist() == pytest.approx(
        [1.0, 0.5, 0.5, np.sqrt(0.3)]
    )
    assert evidence.labels[1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_real_evidence_loader_rejects_known_corrupt_source(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    Image.fromarray(np.zeros((48, 64), dtype=np.uint8)).save(corrupt / "frame.png")
    manifest = tmp_path / "door-labels.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "samples": [
                    {
                        "frame": "corrupt/frame.png",
                        "visible": False,
                    }
                ],
            }
        )
    )
    registry = load_frame_integrity_registry(
        _write_integrity_registry(tmp_path),
        root=tmp_path,
    )

    with pytest.raises(FrameIntegrityError, match="known_corrupt"):
        load_real_door_evidence(
            manifest,
            root=tmp_path,
            integrity_registry=registry,
        )
