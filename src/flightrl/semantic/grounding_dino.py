from __future__ import annotations

from dataclasses import dataclass
import os
from time import perf_counter
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps

from .contract import GroundingDetection, GroundingResult, NormalizedBox
from .model_artifact import (
    WeightsFormat,
    huggingface_model_source,
    validate_huggingface_weights_format,
    validate_optional_huggingface_snapshot,
)


class DetectionVerifier(Protocol):
    def verify(
        self,
        image: Image.Image,
        prompt: str,
        detections: tuple[GroundingDetection, ...],
    ) -> tuple[GroundingDetection, ...]: ...


@dataclass(frozen=True, slots=True)
class GroundingDinoConfig:
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    revision: str | None = None
    artifact_manifest: tuple[tuple[str, str], ...] = ()
    runtime_versions: tuple[tuple[str, str], ...] = ()
    weights_format: WeightsFormat | None = None
    device: str = "cpu"
    threshold: float = 0.25
    autocontrast: bool = True
    minimum_box_area: float = 0.0005
    maximum_box_area: float = 0.5
    distractor_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_optional_huggingface_snapshot(
            model_id=self.model_id,
            revision=self.revision,
            manifest=self.artifact_manifest,
            runtime_versions=self.runtime_versions,
        )
        validate_huggingface_weights_format(
            revision=self.revision,
            manifest=self.artifact_manifest,
            weights_format=self.weights_format,
        )
        if self.device not in {"cpu", "mps"}:
            raise ValueError("Grounding DINO device must be cpu or mps")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("Grounding DINO threshold must be in (0, 1)")
        if not 0.0 < self.minimum_box_area < self.maximum_box_area <= 1.0:
            raise ValueError("Grounding DINO box area limits are invalid")


class GroundingDinoGrounder:
    def __init__(
        self,
        config: GroundingDinoConfig | None = None,
        *,
        verifier: DetectionVerifier | None = None,
    ) -> None:
        self.config = config or GroundingDinoConfig()
        self.verifier = verifier
        os.environ.setdefault("USE_TF", "0")
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "semantic grounding requires: python -m pip install -e '.[semantic]'"
            ) from exc
        if self.config.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS grounding requested but torch reports MPS unavailable")
        self.torch = torch
        with huggingface_model_source(
            model_id=self.config.model_id,
            revision=self.config.revision,
            manifest=self.config.artifact_manifest,
            runtime_versions=self.config.runtime_versions,
        ) as model_source:
            self.processor = AutoProcessor.from_pretrained(
                model_source,
                local_files_only=True,
                trust_remote_code=False,
            )
            model_options = {
                "local_files_only": True,
                "trust_remote_code": False,
                "weights_only": True,
            }
            if self.config.weights_format is not None:
                model_options["use_safetensors"] = (
                    self.config.weights_format == "safetensors"
                )
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_source,
                **model_options,
            ).to(self.config.device)
        self.model.eval()

    def detect(
        self,
        pixels: np.ndarray,
        prompt: str,
        *,
        frame_index: int,
        frame_host_time_s: float,
    ) -> GroundingResult:
        source = np.asarray(pixels)
        image = _prepare_image(source, autocontrast=self.config.autocontrast)
        target = prompt.strip()
        labels = list(dict.fromkeys((target, *self.config.distractor_labels)))
        inputs = self.processor(text=labels, images=image, return_tensors="pt")
        inputs = inputs.to(self.config.device)
        started = perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.config.threshold,
            target_sizes=[(image.height, image.width)],
            text_labels=labels,
        )[0]
        detections: list[GroundingDetection] = []
        for label, score, box in _candidate_rows(
            processed,
            target=target,
            target_only=not self.config.distractor_labels,
        ):
            detection = _detection(label, score, box, image.width, image.height)
            if (
                detection is not None
                and self.config.minimum_box_area <= detection.box.area <= self.config.maximum_box_area
                and _matches_target(detection.label, target)
            ):
                detections.append(detection)
        verified = tuple(detections)
        if self.verifier is not None:
            verified = self.verifier.verify(image, prompt, verified)
        inference_ms = (perf_counter() - started) * 1000.0
        return GroundingResult(
            prompt=prompt,
            frame_index=frame_index,
            frame_host_time_s=frame_host_time_s,
            image_width=int(source.shape[1]),
            image_height=int(source.shape[0]),
            source_mean=float(source.mean()),
            inference_ms=inference_ms,
            detections=verified,
            proposed_detections=tuple(detections),
        )


def _prepare_image(pixels: np.ndarray, *, autocontrast: bool) -> Image.Image:
    if pixels.ndim == 2:
        image = Image.fromarray(pixels.astype(np.uint8), mode="L")
        if autocontrast:
            image = ImageOps.autocontrast(image)
        return image.convert("RGB")
    if pixels.ndim == 3 and pixels.shape[2] in (3, 4):
        image = Image.fromarray(pixels.astype(np.uint8)).convert("RGB")
        return ImageOps.autocontrast(image) if autocontrast else image
    raise ValueError(f"unsupported grounding frame shape {pixels.shape}")


def _candidate_rows(
    processed: dict,
    *,
    target: str,
    target_only: bool,
) -> tuple[tuple[object, object, object], ...]:
    scores = processed["scores"]
    boxes = processed["boxes"]
    if target_only:
        return tuple(
            (target, score, box)
            for score, box in zip(scores, boxes, strict=True)
        )
    return tuple(
        zip(processed["text_labels"], scores, boxes, strict=True)
    )


def _detection(
    label,
    score,
    box,
    width: int,
    height: int,
) -> GroundingDetection | None:
    text = str(label).strip()
    if not text:
        return None
    values = [float(value) for value in box]
    x_min = max(0.0, min(1.0, values[0] / width))
    y_min = max(0.0, min(1.0, values[1] / height))
    x_max = max(0.0, min(1.0, values[2] / width))
    y_max = max(0.0, min(1.0, values[3] / height))
    if x_min >= x_max or y_min >= y_max:
        return None
    return GroundingDetection(
        label=text,
        confidence=float(score),
        box=NormalizedBox(x_min, y_min, x_max, y_max),
    )


def _matches_target(label: str, target: str) -> bool:
    return " ".join(label.casefold().split()) == " ".join(target.casefold().split())
