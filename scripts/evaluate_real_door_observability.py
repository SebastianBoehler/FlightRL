from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flightrl.semantic.door_calibration import temporal_calibration_split
from flightrl.semantic.door_observability import (
    DoorObservabilityGate,
    DoorObservabilityMetrics,
    decode_observability,
    door_observability_model_from_state,
    observability_metrics,
)
from flightrl.semantic.door_real_evidence import load_real_door_evidence
from flightrl.semantic.frame_integrity import load_frame_integrity_registry


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    source_report = json.loads(args.synthetic_report.read_text())
    synthetic = _synthetic_metrics(source_report)
    registry = load_frame_integrity_registry(
        args.integrity_registry,
        root=ROOT,
    )
    evidence = load_real_door_evidence(
        args.real_manifest,
        root=ROOT,
        integrity_registry=registry,
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = _grounder_state(payload)
    model = door_observability_model_from_state(state)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        predictions = decode_observability(
            model(torch.from_numpy(evidence.frames))
        ).numpy()
    positive = evidence.labels[:, 0] > 0.5
    negative = ~positive
    split = temporal_calibration_split(predictions[:, 0], evidence.labels[:, 0])
    positive_metrics = _subset_metrics(
        predictions,
        evidence.labels,
        positive & split.evaluation_mask,
        threshold=split.threshold,
    )
    negative_metrics = _subset_metrics(
        predictions,
        evidence.labels,
        negative & split.evaluation_mask,
        threshold=split.threshold,
    )
    gate = DoorObservabilityGate()
    result = gate.evaluate(
        synthetic=synthetic,
        real_positive=positive_metrics,
        real_negative=negative_metrics,
    )
    report = {
        "schema_version": 1,
        "experiment": "fixed_door_real_observability_gate",
        "status": result.status,
        "checkpoint": str(args.checkpoint.resolve()),
        "synthetic_report": str(args.synthetic_report.resolve()),
        "real_manifest": str(args.real_manifest.resolve()),
        "sample_count": int(evidence.labels.shape[0]),
        "calibration": {
            "method": "first temporal third per class",
            "sample_count": int(np.sum(split.calibration_mask)),
            "positive_count": int(np.sum(positive & split.calibration_mask)),
            "negative_count": int(np.sum(negative & split.calibration_mask)),
            "visibility_threshold": split.threshold,
        },
        "held_out_sample_count": int(np.sum(split.evaluation_mask)),
        "positive_metrics": positive_metrics.to_dict(),
        "negative_metrics": negative_metrics.to_dict(),
        "gate": result.to_dict(),
        "thresholds": {
            "min_visibility_auroc": gate.min_visibility_auroc,
            "max_centroid_error_widths": gate.max_centroid_error_widths,
            "min_real_recall": gate.min_real_recall,
            "max_real_false_positive_rate": gate.max_real_false_positive_rate,
        },
        "prediction_summary": {
            "positive_score_median": float(np.median(predictions[positive, 0])),
            "negative_score_median": float(np.median(predictions[negative, 0])),
            "positive_score_range": [
                float(np.min(predictions[positive, 0])),
                float(np.max(predictions[positive, 0])),
            ],
            "negative_score_range": [
                float(np.min(predictions[negative, 0])),
                float(np.max(predictions[negative, 0])),
            ],
        },
        "errors": {
            "false_positive_frames": _error_frames(
                evidence.frame_paths,
                split.evaluation_mask & negative,
                predictions[:, 0] >= split.threshold,
            ),
            "false_negative_frames": _error_frames(
                evidence.frame_paths,
                split.evaluation_mask & positive,
                predictions[:, 0] < split.threshold,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def _error_frames(
    paths: tuple[Path, ...],
    eligible: np.ndarray,
    failed: np.ndarray,
) -> list[str]:
    return [
        str(path)
        for path, selected in zip(paths, eligible & failed, strict=True)
        if selected
    ]


def _grounder_state(payload: dict) -> dict:
    source = payload.get("state_dict", payload)
    prefix = "encoder.grounder."
    extracted = {
        key[len(prefix) :]: value
        for key, value in source.items()
        if key.startswith(prefix)
    }
    return extracted or source


def _synthetic_metrics(report: dict) -> DoorObservabilityMetrics:
    if "synthetic_metrics" in report:
        return DoorObservabilityMetrics(**report["synthetic_metrics"])
    metrics = report["bootstrap"]["grounder"]["evaluation"]
    if "native" in metrics:
        metrics = metrics["native"]
    sample_count = int(report.get("grounder_evaluation_samples", 16_384))
    positive_count = round(sample_count * metrics["positive_fraction"])
    return DoorObservabilityMetrics(
        sample_count=sample_count,
        positive_count=positive_count,
        negative_count=sample_count - positive_count,
        visibility_auroc=metrics["visibility_auroc"],
        visibility_recall=metrics["visibility_recall"],
        false_positive_rate=metrics["visibility_false_positive_rate"],
        centroid_median_error_widths=metrics["centroid_median_error_widths"],
    )


def _subset_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: float,
) -> DoorObservabilityMetrics:
    if not np.any(mask):
        raise ValueError("real evidence must contain both positive and negative samples")
    return observability_metrics(
        predictions[mask],
        labels[mask],
        visibility_threshold=threshold,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a synthetic-pass door head on reviewed real frames."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "artifacts/semantic/door-observability-64x48-r128-20260729"
        / "door_observability.pt",
    )
    parser.add_argument(
        "--synthetic-report",
        type=Path,
        default=ROOT
        / "artifacts/semantic/door-observability-64x48-r128-20260729"
        / "report.json",
    )
    parser.add_argument(
        "--real-manifest",
        type=Path,
        default=ROOT / "configs/semantic/door_observability_real_20260729.json",
    )
    parser.add_argument(
        "--integrity-registry",
        type=Path,
        default=ROOT / "configs/semantic/aideck_frame_integrity.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts/semantic/door-observability-real-gate-20260729"
        / "report.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
