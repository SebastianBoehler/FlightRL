from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .frame_integrity import FrameIntegrityRegistry


@dataclass(frozen=True, slots=True)
class RealDoorEvidence:
    frames: np.ndarray
    labels: np.ndarray
    frame_paths: tuple[Path, ...]


def load_real_door_evidence(
    manifest_path: str | Path,
    *,
    root: str | Path,
    integrity_registry: FrameIntegrityRegistry,
) -> RealDoorEvidence:
    payload = json.loads(Path(manifest_path).read_text())
    if payload.get("version") != 1:
        raise ValueError("real door evidence manifest version must be 1")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("real door evidence manifest requires samples")
    base = Path(root).resolve()
    frames: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[Path] = []
    for sample in samples:
        frame_path, label = _parse_sample(sample, base)
        integrity_registry.require_frame_safe(frame_path.parent)
        frames.append(load_real_door_frame(frame_path))
        labels.append(label)
        paths.append(frame_path)
    return RealDoorEvidence(
        frames=np.stack(frames).astype(np.float32)[:, None, ...],
        labels=np.stack(labels).astype(np.float32),
        frame_paths=tuple(paths),
    )


def _parse_sample(
    sample: object,
    root: Path,
) -> tuple[Path, np.ndarray]:
    if not isinstance(sample, dict):
        raise ValueError("real door samples must be objects")
    relative_path = Path(str(sample.get("frame", "")))
    if not str(relative_path) or relative_path.is_absolute():
        raise ValueError("real door frame paths must be non-empty and relative")
    frame_path = (root / relative_path).resolve()
    if root not in frame_path.parents:
        raise ValueError("real door frame path escapes the dataset root")
    visible = sample.get("visible")
    if not isinstance(visible, bool):
        raise ValueError("real door sample visibility must be boolean")
    if not visible:
        if "box" in sample:
            raise ValueError("door-negative samples cannot define a box")
        return frame_path, np.zeros(4, dtype=np.float32)
    box = sample.get("box")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("door-positive samples require a normalized four-value box")
    x_min, y_min, x_max, y_max = (float(value) for value in box)
    if not (
        0.0 <= x_min < x_max <= 1.0
        and 0.0 <= y_min < y_max <= 1.0
    ):
        raise ValueError("real door boxes must be ordered within [0, 1]")
    return frame_path, np.asarray(
        (
            1.0,
            0.5 * (x_min + x_max),
            0.5 * (y_min + y_max),
            np.sqrt((x_max - x_min) * (y_max - y_min)),
        ),
        dtype=np.float32,
    )


def load_real_door_frame(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L").resize((64, 48), Image.Resampling.BILINEAR)
        pixels = np.asarray(gray, dtype=np.float32)
    quantized = np.rint(pixels / 17.0) * 17.0
    return np.clip(quantized, 0.0, 255.0).astype(np.float32) / 255.0
