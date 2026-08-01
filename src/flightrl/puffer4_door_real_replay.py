from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from flightrl.semantic.door_calibration import temporal_calibration_split
from flightrl.semantic.door_real_evidence import (
    load_real_door_evidence,
    load_real_door_frame,
)
from flightrl.semantic.frame_integrity import load_frame_integrity_registry


ROOT = Path(__file__).resolve().parents[2]


def load_real_grounding_calibration() -> tuple[torch.Tensor, torch.Tensor]:
    registry = load_frame_integrity_registry(
        ROOT / "configs/semantic/aideck_frame_integrity.json",
        root=ROOT,
    )
    evidence = load_real_door_evidence(
        ROOT / "configs/semantic/door_observability_real_20260729.json",
        root=ROOT,
        integrity_registry=registry,
    )
    split = temporal_calibration_split(
        np.zeros(evidence.labels.shape[0]),
        evidence.labels[:, 0],
    )
    positive = (
        split.calibration_mask
        & (evidence.labels[:, 0] > 0.5)
    )
    negative_dir = (
        ROOT
        / "artifacts/ai_deck/door-observability-negative-20260729-run1/frames"
    )
    registry.require_frame_safe(negative_dir)
    negative_paths = sorted(negative_dir.glob("frame-*.png"))
    negative_frames = np.stack(
        [load_real_door_frame(path) for path in negative_paths[: len(negative_paths) // 3]]
    )[:, None, ...]
    frames = np.concatenate((evidence.frames[positive], negative_frames))
    labels = np.concatenate(
        (
            evidence.labels[positive],
            np.zeros((negative_frames.shape[0], 4), dtype=np.float32),
        )
    )
    return torch.from_numpy(frames), torch.from_numpy(labels)
