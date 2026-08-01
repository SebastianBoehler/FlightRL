from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Iterable

from PIL import Image

from .contract import GroundingDetection
from .model_artifact import (
    WeightsFormat,
    huggingface_model_source,
    validate_huggingface_weights_format,
    validate_optional_huggingface_snapshot,
)


DEFAULT_NEGATIVE_DESCRIPTIONS = (
    "an oven",
    "a window",
    "a kitchen cabinet",
    "a wall",
    "background or an unclear object",
)


@dataclass(frozen=True, slots=True)
class ClipVerifierConfig:
    model_id: str = "openai/clip-vit-base-patch32"
    revision: str | None = None
    artifact_manifest: tuple[tuple[str, str], ...] = ()
    runtime_versions: tuple[tuple[str, str], ...] = ()
    weights_format: WeightsFormat | None = None
    device: str = "mps"
    minimum_probability: float = 0.60
    minimum_margin: float = 0.45
    crop_padding: float = 0.6
    negative_descriptions: tuple[str, ...] = DEFAULT_NEGATIVE_DESCRIPTIONS

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
            raise ValueError("CLIP verifier device must be cpu or mps")
        if not 0.0 < self.minimum_probability < 1.0:
            raise ValueError("CLIP minimum probability must be in (0, 1)")
        if not 0.0 <= self.minimum_margin < 1.0:
            raise ValueError("CLIP minimum margin must be in [0, 1)")
        if not 0.0 <= self.crop_padding <= 2.0:
            raise ValueError("CLIP crop padding must be in [0, 2]")


class ClipCropVerifier:
    def __init__(self, config: ClipVerifierConfig | None = None) -> None:
        self.config = config or ClipVerifierConfig()
        os.environ.setdefault("USE_TF", "0")
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "CLIP verification requires: python -m pip install -e '.[semantic]'"
            ) from exc
        if self.config.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS CLIP verification requested but unavailable")
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
            self.model = CLIPModel.from_pretrained(
                model_source,
                **model_options,
            ).to(self.config.device)
        self.model.eval()

    def verify(
        self,
        image: Image.Image,
        prompt: str,
        detections: Iterable[GroundingDetection],
    ) -> tuple[GroundingDetection, ...]:
        candidates = tuple(detections)
        if not candidates:
            return ()
        descriptions = (
            target_description(prompt),
            *self.config.negative_descriptions,
        )
        crops = [
            padded_crop(image, detection, self.config.crop_padding)
            for detection in candidates
        ]
        inputs = self.processor(
            text=list(descriptions),
            images=crops,
            return_tensors="pt",
            padding=True,
        ).to(self.config.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        probabilities = outputs.logits_per_image.softmax(dim=1).cpu().tolist()
        verified: list[GroundingDetection] = []
        for detection, scores in zip(candidates, probabilities, strict=True):
            target_probability = float(scores[0])
            strongest_negative = max(float(value) for value in scores[1:])
            margin = target_probability - strongest_negative
            if (
                target_probability >= self.config.minimum_probability
                and margin >= self.config.minimum_margin
            ):
                verified.append(
                    replace(
                        detection,
                        verification_confidence=target_probability,
                        verification_margin=margin,
                    )
                )
        return tuple(verified)


def target_description(prompt: str) -> str:
    normalized = " ".join(prompt.casefold().split())
    if normalized in {"computer monitor", "monitor", "screen"}:
        return "a computer monitor on a desk"
    if normalized == "window":
        return "a window in a room"
    return f"a photo of {prompt.strip()}"


def padded_crop(
    image: Image.Image,
    detection: GroundingDetection,
    padding: float,
) -> Image.Image:
    box = detection.box
    box_width = box.x_max - box.x_min
    box_height = box.y_max - box.y_min
    x_min = max(0.0, box.x_min - padding * box_width)
    y_min = max(0.0, box.y_min - padding * box_height)
    x_max = min(1.0, box.x_max + padding * box_width)
    y_max = min(1.0, box.y_max + padding * box_height)
    width, height = image.size
    return image.crop(
        (
            round(x_min * width),
            round(y_min * height),
            round(x_max * width),
            round(y_max * height),
        )
    )
