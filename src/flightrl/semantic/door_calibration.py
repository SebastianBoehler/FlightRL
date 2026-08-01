from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DoorCalibrationSplit:
    threshold: float
    calibration_mask: np.ndarray
    evaluation_mask: np.ndarray


def temporal_calibration_split(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    fraction: float = 1.0 / 3.0,
) -> DoorCalibrationSplit:
    confidence = np.asarray(scores, dtype=np.float64)
    visible = np.asarray(labels, dtype=np.float64) > 0.5
    if confidence.ndim != 1 or visible.shape != confidence.shape:
        raise ValueError("scores and labels must be matching one-dimensional arrays")
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration fraction must be between zero and one")
    positive_indices = np.flatnonzero(visible)
    negative_indices = np.flatnonzero(~visible)
    if positive_indices.size < 3 or negative_indices.size < 3:
        raise ValueError("calibration requires at least three samples per class")
    positive_count = _calibration_count(positive_indices.size, fraction)
    negative_count = _calibration_count(negative_indices.size, fraction)
    calibration_mask = np.zeros(confidence.shape, dtype=bool)
    calibration_mask[positive_indices[:positive_count]] = True
    calibration_mask[negative_indices[:negative_count]] = True
    calibration_positive = confidence[calibration_mask & visible]
    calibration_negative = confidence[calibration_mask & ~visible]
    threshold = 0.5 * (
        float(np.min(calibration_positive))
        + float(np.max(calibration_negative))
    )
    return DoorCalibrationSplit(
        threshold=threshold,
        calibration_mask=calibration_mask,
        evaluation_mask=~calibration_mask,
    )


def _calibration_count(sample_count: int, fraction: float) -> int:
    return min(sample_count - 1, max(1, int(np.floor(sample_count * fraction))))
