from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


SCHEMA = "flightrl.aideck_paired_observability.v1"
MINIMUM_BALANCED_ACCURACY = 0.95
MINIMUM_CLASSIFICATION_MARGIN = 0.10


def evaluate_paired_gray4(
    positive_frames: np.ndarray,
    negative_frames: np.ndarray,
    *,
    positive_indices: list[int],
    negative_indices: list[int],
    positive_source: Path,
    negative_source: Path,
    positive_metadata: dict[str, object],
    negative_metadata: dict[str, object],
) -> dict[str, object]:
    positive = _structural_features(positive_frames, "positive")
    negative = _structural_features(negative_frames, "negative")
    if len(positive) != len(negative) or len(positive) < 4 or len(positive) % 2:
        raise ValueError("paired gate requires equal even sample counts of at least four")
    if len(positive_indices) != len(positive) or len(negative_indices) != len(negative):
        raise ValueError("paired gate frame indices do not match sampled frames")

    metrics = _two_fold_metrics(positive, negative)
    histogram_metrics = _two_fold_metrics(
        _histogram_features(positive_frames),
        _histogram_features(negative_frames),
    )
    scene_confound_detected = (
        histogram_metrics["balanced_accuracy"] >= MINIMUM_BALANCED_ACCURACY
    )
    return {
        "schema": SCHEMA,
        "method": "two_fold_contiguous_nearest_centroid_after_per_frame_centering_and_l2_normalization",
        "input_contract": {
            "shape": [48, 64],
            "dtype": "uint8",
            "levels": "decoded_gray4_multiples_of_17",
            "model_normalization": "float32(nibble) / 15",
        },
        "positive": _source_record(positive_source, positive_indices, positive_metadata),
        "negative": _source_record(negative_source, negative_indices, negative_metadata),
        **metrics,
        "nonspatial_histogram_baseline": {
            "method": "normalized_16_bin_gray4_histogram_without_spatial_layout",
            **histogram_metrics,
        },
        "spatial_balanced_accuracy_gain": (
            metrics["balanced_accuracy"] - histogram_metrics["balanced_accuracy"]
        ),
        "scene_confound_detected": scene_confound_detected,
        "criteria": {
            "minimum_balanced_accuracy": MINIMUM_BALANCED_ACCURACY,
            "minimum_classification_margin": MINIMUM_CLASSIFICATION_MARGIN,
        },
        "paired_observability_passed": _metrics_pass(metrics),
        "shortcut_resistant_semantic_calibration_passed": (
            _metrics_pass(metrics) and not scene_confound_detected
        ),
        "training_authority": False,
        "semantic_generalization_authority": False,
        "deployment_authority": False,
        "authority_reason": (
            "This calibrates separability of two operator-labeled stationary scenes only. "
            "A passing nonspatial histogram baseline identifies a scene/exposure confound; "
            "neither result establishes door-category generalization, capture integrity, "
            "telemetry synchronization, training eligibility, or control authority."
        ),
    }


def _two_fold_metrics(
    positive: np.ndarray, negative: np.ndarray
) -> dict[str, float]:
    predictions: list[tuple[int, bool, float]] = []
    midpoint = len(positive) // 2
    for training, evaluation in (
        (slice(0, midpoint), slice(midpoint, None)),
        (slice(midpoint, None), slice(0, midpoint)),
    ):
        positive_centroid = _unit_centroid(positive[training])
        negative_centroid = _unit_centroid(negative[training])
        predictions.extend(
            _classify(positive[evaluation], 1, positive_centroid, negative_centroid)
        )
        predictions.extend(
            _classify(negative[evaluation], 0, positive_centroid, negative_centroid)
        )
    positive_recall = _recall(predictions, 1)
    negative_recall = _recall(predictions, 0)
    margins = [item[2] for item in predictions]
    return {
        "balanced_accuracy": (positive_recall + negative_recall) / 2.0,
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "minimum_classification_margin": min(margins),
        "median_classification_margin": float(np.median(margins)),
    }


def _metrics_pass(metrics: dict[str, float]) -> bool:
    return (
        metrics["balanced_accuracy"] >= MINIMUM_BALANCED_ACCURACY
        and metrics["minimum_classification_margin"]
        >= MINIMUM_CLASSIFICATION_MARGIN
    )


def _histogram_features(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames)
    histograms = np.stack(
        [
            np.bincount((frame.reshape(-1) // 17), minlength=16).astype(np.float32)
            for frame in values
        ]
    )
    histograms /= histograms.sum(axis=1, keepdims=True)
    return histograms / np.linalg.norm(histograms, axis=1, keepdims=True)


def _structural_features(frames: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(frames)
    if values.ndim != 3 or values.shape[1:] != (48, 64) or values.dtype != np.uint8:
        raise ValueError(f"{label} capture must be [frames, 48, 64] uint8")
    if np.any(values % 17 != 0):
        raise ValueError(f"{label} capture must contain exact decoded gray4 levels")
    features = values.astype(np.float32).reshape(len(values), -1) / 255.0
    features -= features.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError(f"{label} capture contains a structurally empty frame")
    return features / norms


def _unit_centroid(features: np.ndarray) -> np.ndarray:
    centroid = features.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm <= 1.0e-12:
        raise ValueError("paired gate calibration centroid is degenerate")
    return centroid / norm


def _classify(
    features: np.ndarray,
    label: int,
    positive_centroid: np.ndarray,
    negative_centroid: np.ndarray,
) -> list[tuple[int, bool, float]]:
    positive_distance = 1.0 - features @ positive_centroid
    negative_distance = 1.0 - features @ negative_centroid
    predictions = positive_distance < negative_distance
    margins = np.where(
        label == 1,
        negative_distance - positive_distance,
        positive_distance - negative_distance,
    )
    return [
        (label, bool(prediction == label), float(margin))
        for prediction, margin in zip(predictions, margins, strict=True)
    ]


def _recall(predictions: list[tuple[int, bool, float]], label: int) -> float:
    matches = [correct for expected, correct, _margin in predictions if expected == label]
    return sum(matches) / len(matches)


def _source_record(
    source: Path,
    indices: list[int],
    metadata: dict[str, object],
) -> dict[str, object]:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "path": str(source),
        "sha256": digest,
        "sampled_frame_indices": indices,
        "capture_metadata": metadata,
    }
