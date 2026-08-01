from __future__ import annotations

import json
from pathlib import Path

import pytest

from flightrl.semantic.frame_integrity import (
    FrameIntegrityError,
    load_frame_integrity_registry,
)


def _write_registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "datasets": [
                    {
                        "path": "artifacts/frame-safe/frames",
                        "status": "frame_safe",
                        "evidence": "moving capture reviewed after the three-buffer fix",
                    },
                    {
                        "path": "artifacts/raster-wrap/frames",
                        "status": "known_corrupt",
                        "evidence": "two sensor frames appear in one decoded image",
                    },
                ],
            }
        )
    )


def test_registry_accepts_only_explicit_frame_safe_datasets(tmp_path: Path) -> None:
    registry_path = tmp_path / "integrity.json"
    _write_registry(registry_path)
    registry = load_frame_integrity_registry(registry_path, root=tmp_path)

    record = registry.require_frame_safe(tmp_path / "artifacts/frame-safe/frames")

    assert record.status == "frame_safe"
    assert "three-buffer" in record.evidence


def test_registry_rejects_corrupt_and_unreviewed_datasets(tmp_path: Path) -> None:
    registry_path = tmp_path / "integrity.json"
    _write_registry(registry_path)
    registry = load_frame_integrity_registry(registry_path, root=tmp_path)

    with pytest.raises(FrameIntegrityError, match="known_corrupt"):
        registry.require_frame_safe(tmp_path / "artifacts/raster-wrap/frames")
    with pytest.raises(FrameIntegrityError, match="not registered"):
        registry.require_frame_safe(tmp_path / "artifacts/unknown/frames")


def test_registry_rejects_invalid_status(tmp_path: Path) -> None:
    registry_path = tmp_path / "integrity.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "datasets": [
                    {
                        "path": "frames",
                        "status": "probably_ok",
                        "evidence": "guess",
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="status"):
        load_frame_integrity_registry(registry_path, root=tmp_path)
