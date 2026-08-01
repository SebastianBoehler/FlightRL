from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import time

import numpy as np
from PIL import Image

from flightrl.hardware.aideck_stream import AiDeckFrame

from .contract import GroundingDetection, GroundingResult
from .dataset import SemanticRunWriter
from .grounding_dino import GroundingDinoGrounder


@dataclass(frozen=True, slots=True)
class ResolutionVariant:
    width: int
    height: int
    bits: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("resolution dimensions must be positive")
        if self.bits not in {4, 8}:
            raise ValueError("resolution sweep supports 4-bit or 8-bit grayscale")

    @property
    def name(self) -> str:
        return f"{self.width}x{self.height}-gray{self.bits}"


@dataclass(frozen=True, slots=True)
class EvaluatedFrame:
    result: GroundingResult
    best: GroundingDetection | None


def degrade_frame(pixels: np.ndarray, variant: ResolutionVariant) -> np.ndarray:
    image = Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="L")
    resized = np.asarray(
        image.resize((variant.width, variant.height), Image.Resampling.BOX)
    ).copy()
    if variant.bits == 8:
        return resized
    levels = (1 << variant.bits) - 1
    quantized = np.rint(resized.astype(np.float32) * levels / 255.0)
    return np.rint(quantized * 255.0 / levels).astype(np.uint8)


def evaluate_variant(
    paths: list[Path],
    *,
    prompt: str,
    grounder: GroundingDinoGrounder,
    variant: ResolutionVariant,
    output_dir: Path,
) -> list[EvaluatedFrame]:
    manifest = {
        "mode": "resolution_sweep",
        "prompt": prompt,
        "variant": variant.name,
        "sources": [str(path) for path in paths],
    }
    evaluated: list[EvaluatedFrame] = []
    with SemanticRunWriter(output_dir / variant.name, manifest=manifest) as writer:
        for index, path in enumerate(paths, start=1):
            source = np.asarray(Image.open(path).convert("L"))
            pixels = degrade_frame(source, variant)
            frame = AiDeckFrame(
                index,
                time(),
                variant.width,
                variant.height,
                1,
                1,
                pixels,
            )
            result = grounder.detect(
                pixels,
                prompt,
                frame_index=index,
                frame_host_time_s=frame.host_time_s,
            )
            writer.write(frame, result)
            evaluated.append(EvaluatedFrame(result, result.best))
    return evaluated


def variant_metrics(
    evaluated: list[EvaluatedFrame],
    baseline: list[EvaluatedFrame],
) -> dict[str, float | int | bool | None]:
    if len(evaluated) != len(baseline) or not evaluated:
        raise ValueError("variant and baseline evaluations must be non-empty and aligned")
    baseline_positive = sum(item.best is not None for item in baseline)
    baseline_negative = len(baseline) - baseline_positive
    retained = 0
    new_false_positives = 0
    overlaps: list[float] = []
    confidences: list[float] = []
    inference_ms: list[float] = []
    for current, reference in zip(evaluated, baseline, strict=True):
        inference_ms.append(current.result.inference_ms)
        if current.best is not None:
            confidences.append(current.best.confidence)
        if reference.best is not None and current.best is not None:
            retained += 1
            overlaps.append(
                max(
                    box_iou(reference_detection, current_detection)
                    for reference_detection in reference.result.detections
                    for current_detection in current.result.detections
                )
            )
        elif reference.best is None and current.best is not None:
            new_false_positives += 1
    baseline_recall = retained / baseline_positive if baseline_positive else 0.0
    false_positive_rate = (
        new_false_positives / baseline_negative if baseline_negative else 0.0
    )
    median_iou = median(overlaps) if overlaps else 0.0
    return {
        "frames": len(evaluated),
        "frames_with_detection": len(confidences),
        "detection_rate": len(confidences) / len(evaluated),
        "baseline_recall": baseline_recall,
        "new_false_positive_rate": false_positive_rate,
        "median_box_iou": median_iou,
        "median_confidence": median(confidences) if confidences else None,
        "median_inference_ms": median(inference_ms),
        "signal_retained": (
            baseline_positive > 0
            and baseline_recall >= 0.8
            and median_iou >= 0.5
            and false_positive_rate <= 0.1
        ),
    }


def box_iou(first: GroundingDetection, second: GroundingDetection) -> float:
    a = first.box
    b = second.box
    intersection_width = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    intersection_height = max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    intersection = intersection_width * intersection_height
    union = a.area + b.area - intersection
    return intersection / union if union > 0.0 else 0.0
