from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("normalized box values must be finite and in [0, 1]")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("normalized box must have positive width and height")

    @property
    def center_x(self) -> float:
        return 0.5 * (self.x_min + self.x_max)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.y_min + self.y_max)

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


@dataclass(frozen=True, slots=True)
class GroundingDetection:
    label: str
    confidence: float
    box: NormalizedBox
    verification_confidence: float | None = None
    verification_margin: float | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("grounding label cannot be empty")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("grounding confidence must be in [0, 1]")
        optional_values = (
            self.verification_confidence,
            self.verification_margin,
        )
        if any(value is not None and not isfinite(value) for value in optional_values):
            raise ValueError("verification scores must be finite when provided")
        if (
            self.verification_confidence is not None
            and not 0.0 <= self.verification_confidence <= 1.0
        ):
            raise ValueError("verification confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class GroundingResult:
    prompt: str
    frame_index: int
    frame_host_time_s: float
    image_width: int
    image_height: int
    source_mean: float
    inference_ms: float
    detections: tuple[GroundingDetection, ...]
    proposed_detections: tuple[GroundingDetection, ...] = ()

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("grounding prompt cannot be empty")
        if self.frame_index < 0 or self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("grounding frame metadata must be positive")
        if not all(
            isfinite(value)
            for value in (self.frame_host_time_s, self.source_mean, self.inference_ms)
        ):
            raise ValueError("grounding timing and image statistics must be finite")

    @property
    def best(self) -> GroundingDetection | None:
        return max(self.detections, key=lambda detection: detection.confidence, default=None)

    @property
    def best_proposal(self) -> GroundingDetection | None:
        proposals = self.proposed_detections or self.detections
        return max(proposals, key=lambda detection: detection.confidence, default=None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
