from __future__ import annotations

from math import isfinite, sqrt

import numpy as np

from flightrl.semantic.contract import GroundingDetection


DOOR_EVIDENCE_DIM = 5


def detector_evidence(
    detection: GroundingDetection | None,
    *,
    age_s: float | None,
    maximum_age_s: float,
) -> np.ndarray:
    """Encode host detector output as confidence, geometry, and normalized age."""
    if maximum_age_s <= 0.0:
        raise ValueError("maximum_age_s must be positive")
    stale = age_s is None or not isfinite(age_s) or age_s >= maximum_age_s
    age = 1.0 if stale else np.clip(age_s / maximum_age_s, 0.0, 1.0)
    detection = None if stale else detection
    if detection is None:
        return np.asarray((0.0, 0.0, 0.0, 0.0, age), dtype=np.float32)
    confidence = (
        detection.verification_confidence
        if detection.verification_confidence is not None
        else detection.confidence
    )
    return np.asarray(
        (
            confidence,
            2.0 * detection.box.center_x - 1.0,
            2.0 * detection.box.center_y - 1.0,
            sqrt(detection.box.area),
            age,
        ),
        dtype=np.float32,
    )


def observable_teacher_action(
    evidence: np.ndarray,
    *,
    target_seen: bool,
    recovery_yaw: float = 0.70,
) -> np.ndarray:
    values = np.asarray(evidence, dtype=np.float32)
    if values.shape != (DOOR_EVIDENCE_DIM,):
        raise ValueError(f"evidence must have shape ({DOOR_EVIDENCE_DIM},)")
    detected = values[0] > 0.0 and values[4] < 1.0
    if not detected:
        return np.asarray(
            (0.0, recovery_yaw if target_seen else 0.85),
            dtype=np.float32,
        )
    yaw = float(np.clip(-1.6 * values[1], -1.0, 1.0))
    centered = abs(float(values[1])) < 0.25
    if not centered or values[3] >= 0.78:
        forward = 0.0
    else:
        forward = float(np.clip(1.6 * (0.78 - values[3]), 0.15, 0.72))
    return np.asarray((forward, yaw), dtype=np.float32)
