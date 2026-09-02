from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import time

import numpy as np
from PIL import Image

from .aideck_pair_gate import evaluate_paired_gray4


@dataclass(frozen=True, slots=True)
class ArchivedFrame:
    source: Path
    index: int
    host_time_s: float
    pixels: np.ndarray
    capture_metadata: dict[str, object]


def image_paths(path: Path, limit: int) -> list[Path]:
    if limit <= 0:
        raise ValueError("--max-frames must be positive")
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    extensions = {".png", ".jpg", ".jpeg"}
    paths = sorted(item for item in path.iterdir() if item.suffix.lower() in extensions)
    if not paths:
        raise FileNotFoundError(f"no PNG/JPEG frames found in {path}")
    if len(paths) <= limit:
        return paths
    indices = np.linspace(0, len(paths) - 1, limit, dtype=int)
    return [paths[index] for index in indices]


def load_archived_frames(path: Path, limit: int) -> list[ArchivedFrame]:
    if path.is_file() and path.suffix.lower() == ".npz":
        return _load_npz_frames(path, limit)
    return [
        ArchivedFrame(
            source=image_path,
            index=index,
            host_time_s=time(),
            pixels=np.asarray(Image.open(image_path).convert("L")),
            capture_metadata={},
        )
        for index, image_path in enumerate(image_paths(path, limit))
    ]


def evaluate_paired_captures(
    positive_path: Path,
    negative_path: Path,
    *,
    sample_count: int,
) -> dict[str, object]:
    positive = load_archived_frames(positive_path, sample_count)
    negative = load_archived_frames(negative_path, sample_count)
    return evaluate_paired_gray4(
        np.stack([frame.pixels for frame in positive]),
        np.stack([frame.pixels for frame in negative]),
        positive_indices=[frame.index for frame in positive],
        negative_indices=[frame.index for frame in negative],
        positive_source=positive_path,
        negative_source=negative_path,
        positive_metadata=positive[0].capture_metadata,
        negative_metadata=negative[0].capture_metadata,
    )


def _load_npz_frames(path: Path, limit: int) -> list[ArchivedFrame]:
    if limit <= 0:
        raise ValueError("--max-frames must be positive")
    with np.load(path, allow_pickle=False) as artifact:
        required = {"decoded_frames", "host_time_s", "metadata_json"}
        if not required.issubset(artifact.files):
            raise ValueError(
                "AI Deck NPZ must contain decoded_frames, host_time_s, and metadata_json"
            )
        frames = np.asarray(artifact["decoded_frames"])
        host_times = np.asarray(artifact["host_time_s"], dtype=np.float64)
        metadata = json.loads(str(artifact["metadata_json"]))
    if frames.ndim != 3 or frames.dtype != np.uint8 or len(frames) == 0:
        raise ValueError(
            "AI Deck decoded_frames must be non-empty [frames, height, width] uint8"
        )
    if not isinstance(metadata, dict):
        raise ValueError("AI Deck metadata_json must decode to an object")
    if (
        host_times.shape != (len(frames),)
        or not np.isfinite(host_times).all()
        or np.any(np.diff(host_times) < 0.0)
    ):
        raise ValueError("AI Deck host_time_s must be finite and nondecreasing")
    count = min(limit, len(frames))
    indices = np.linspace(0, len(frames) - 1, count, dtype=int)
    return [
        ArchivedFrame(
            path,
            int(index),
            float(host_times[index]),
            frames[index].copy(),
            metadata,
        )
        for index in indices
    ]
