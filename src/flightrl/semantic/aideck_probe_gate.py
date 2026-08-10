from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .aideck_pair_gate import (
    MINIMUM_CLASSIFICATION_MARGIN,
    MINIMUM_BALANCED_ACCURACY,
    _source_record,
    _structural_features,
    _unit_centroid,
)


SCHEMA = "flightrl.aideck_cross_capture_observability.v1"
LABELS = {"positive", "negative"}


@dataclass(frozen=True, slots=True)
class LabeledGray4Capture:
    label: str
    frames: np.ndarray
    indices: tuple[int, ...]
    source: Path
    metadata: dict[str, object]


def evaluate_cross_capture_gray4(
    positive_calibration: LabeledGray4Capture,
    negative_calibration: LabeledGray4Capture,
    probes: list[LabeledGray4Capture],
) -> dict[str, object]:
    _validate_capture(positive_calibration, expected_label="positive")
    _validate_capture(negative_calibration, expected_label="negative")
    calibration_sources = {
        positive_calibration.source.resolve(),
        negative_calibration.source.resolve(),
    }
    if len(calibration_sources) != 2:
        raise ValueError("positive and negative calibration captures must be distinct")

    positive_features = _structural_features(
        positive_calibration.frames, "positive calibration"
    )
    negative_features = _structural_features(
        negative_calibration.frames, "negative calibration"
    )
    positive_centroid = _unit_centroid(positive_features)
    negative_centroid = _unit_centroid(negative_features)

    records = []
    seen_sources: set[Path] = set()
    for probe in probes:
        _validate_capture(probe)
        resolved = probe.source.resolve()
        if resolved in calibration_sources or resolved in seen_sources:
            raise ValueError("probe captures must be distinct from calibration and each other")
        seen_sources.add(resolved)
        records.append(_evaluate_probe(probe, positive_centroid, negative_centroid))

    positive_records = [record for record in records if record["label"] == "positive"]
    negative_records = [record for record in records if record["label"] == "negative"]
    positive_passed = bool(positive_records) and all(
        record["probe_passed"] for record in positive_records
    )
    negative_passed = bool(negative_records) and all(
        record["probe_passed"] for record in negative_records
    )
    evaluable = bool(positive_records and negative_records)
    return {
        "schema": SCHEMA,
        "method": "fixed_nearest_centroid_calibration_evaluated_on_distinct_capture_files",
        "calibration": {
            "positive": _capture_record(positive_calibration),
            "negative": _capture_record(negative_calibration),
        },
        "probes": records,
        "criteria": {
            "minimum_expected_recall_per_probe": MINIMUM_BALANCED_ACCURACY,
            "minimum_classification_margin_per_probe": MINIMUM_CLASSIFICATION_MARGIN,
            "required_probe_classes": ["positive", "negative"],
        },
        "positive_probe_passed": positive_passed,
        "negative_probe_passed": negative_passed,
        "cross_capture_gate_evaluable": evaluable,
        "cross_capture_observability_passed": (
            evaluable and positive_passed and negative_passed
        ),
        "training_authority": False,
        "semantic_generalization_authority": False,
        "deployment_authority": False,
        "authority_reason": (
            "Distinct capture files reduce temporal pseudoreplication but do not prove "
            "door-category generalization, scene-confound resistance, capture integrity, "
            "training eligibility, synchronization, or control authority."
        ),
    }


def _evaluate_probe(
    probe: LabeledGray4Capture,
    positive_centroid: np.ndarray,
    negative_centroid: np.ndarray,
) -> dict[str, object]:
    features = _structural_features(probe.frames, f"{probe.label} probe")
    positive_distance = 1.0 - features @ positive_centroid
    negative_distance = 1.0 - features @ negative_centroid
    predicted_positive = positive_distance < negative_distance
    expected_positive = probe.label == "positive"
    correct = predicted_positive == expected_positive
    margins = np.where(
        expected_positive,
        negative_distance - positive_distance,
        positive_distance - negative_distance,
    )
    recall = float(correct.mean())
    minimum_margin = float(margins.min())
    return {
        **_capture_record(probe),
        "label": probe.label,
        "predicted_positive_rate": float(predicted_positive.mean()),
        "expected_recall": recall,
        "minimum_classification_margin": minimum_margin,
        "median_classification_margin": float(np.median(margins)),
        "probe_passed": recall >= MINIMUM_BALANCED_ACCURACY
        and minimum_margin >= MINIMUM_CLASSIFICATION_MARGIN,
    }


def _validate_capture(
    capture: LabeledGray4Capture, *, expected_label: str | None = None
) -> None:
    if capture.label not in LABELS or (
        expected_label is not None and capture.label != expected_label
    ):
        raise ValueError("gray4 capture label must match positive or negative role")
    if len(capture.frames) < 1 or len(capture.indices) != len(capture.frames):
        raise ValueError("gray4 capture frames and indices must be non-empty and aligned")


def _capture_record(capture: LabeledGray4Capture) -> dict[str, object]:
    return _source_record(
        capture.source,
        list(capture.indices),
        capture.metadata,
    )
