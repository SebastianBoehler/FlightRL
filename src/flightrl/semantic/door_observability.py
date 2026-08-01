from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class DoorObservationLabel:
    visible: float
    center_x: float
    center_y: float
    scale: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            (self.visible, self.center_x, self.center_y, self.scale),
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class DoorObservabilityMetrics:
    sample_count: int
    positive_count: int
    negative_count: int
    visibility_auroc: float | None
    visibility_recall: float | None
    false_positive_rate: float | None
    centroid_median_error_widths: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DoorObservabilityGateResult:
    status: str
    synthetic_pass: bool
    real_pass: bool | None
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DoorObservabilityGate:
    min_visibility_auroc: float = 0.90
    max_centroid_error_widths: float = 0.12
    min_real_recall: float = 0.80
    max_real_false_positive_rate: float = 0.10

    def evaluate(
        self,
        *,
        synthetic: DoorObservabilityMetrics,
        real_positive: DoorObservabilityMetrics | None,
        real_negative: DoorObservabilityMetrics | None,
    ) -> DoorObservabilityGateResult:
        failures = list(self._synthetic_failures(synthetic))
        synthetic_pass = not failures
        if not synthetic_pass:
            return DoorObservabilityGateResult(
                status="failed_synthetic",
                synthetic_pass=False,
                real_pass=None,
                failures=tuple(failures),
            )
        if real_positive is None or real_negative is None:
            return DoorObservabilityGateResult(
                status="incomplete_real_evidence",
                synthetic_pass=True,
                real_pass=None,
                failures=("labeled real door-positive and door-negative frames are required",),
            )
        failures.extend(self._real_failures(real_positive, real_negative))
        return DoorObservabilityGateResult(
            status="passed" if not failures else "failed_real",
            synthetic_pass=True,
            real_pass=not failures,
            failures=tuple(failures),
        )

    def _synthetic_failures(
        self,
        metrics: DoorObservabilityMetrics,
    ) -> tuple[str, ...]:
        failures: list[str] = []
        if (
            metrics.visibility_auroc is None
            or metrics.visibility_auroc < self.min_visibility_auroc
        ):
            failures.append(
                f"synthetic visibility AUROC must be >= {self.min_visibility_auroc:.2f}"
            )
        if (
            metrics.centroid_median_error_widths is None
            or metrics.centroid_median_error_widths
            > self.max_centroid_error_widths
        ):
            failures.append(
                "synthetic centroid median error must be <= "
                f"{self.max_centroid_error_widths:.2f} image widths"
            )
        return tuple(failures)

    def _real_failures(
        self,
        positive: DoorObservabilityMetrics,
        negative: DoorObservabilityMetrics,
    ) -> tuple[str, ...]:
        failures: list[str] = []
        if (
            positive.visibility_recall is None
            or positive.visibility_recall < self.min_real_recall
        ):
            failures.append(f"real door recall must be >= {self.min_real_recall:.2f}")
        if (
            positive.centroid_median_error_widths is None
            or positive.centroid_median_error_widths
            > self.max_centroid_error_widths
        ):
            failures.append(
                "real centroid median error must be <= "
                f"{self.max_centroid_error_widths:.2f} image widths"
            )
        if (
            negative.false_positive_rate is None
            or negative.false_positive_rate > self.max_real_false_positive_rate
        ):
            failures.append(
                "real door false-positive rate must be <= "
                f"{self.max_real_false_positive_rate:.2f}"
            )
        return tuple(failures)


class DoorObservabilityNet(nn.Module):
    """Small raw-frame head for visibility, centroid, and apparent scale."""

    def __init__(
        self,
        *,
        pool_shape: tuple[int, int] = (3, 4),
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        if min(pool_shape) <= 0 or hidden_size <= 0:
            raise ValueError("observability model dimensions must be positive")
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(pool_shape),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * pool_shape[0] * pool_shape[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.predict_from_features(self.features(frames))

    def features(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames)

    def predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


def door_observability_model_from_state(
    state_dict: dict[str, torch.Tensor],
) -> DoorObservabilityNet:
    first_head = state_dict["head.0.weight"]
    feature_cells = int(first_head.shape[1]) // 32
    pool_shapes = {12: (3, 4), 48: (6, 8)}
    if feature_cells not in pool_shapes:
        raise ValueError(
            f"unsupported observability feature grid with {feature_cells} cells"
        )
    return DoorObservabilityNet(
        pool_shape=pool_shapes[feature_cells],
        hidden_size=int(first_head.shape[0]),
    )


def labels_from_segmentation(
    segmentation: np.ndarray,
    *,
    target_geom_id: int | tuple[int, ...],
    min_visible_pixels: int = 4,
) -> DoorObservationLabel:
    if segmentation.ndim != 3 or segmentation.shape[2] < 1:
        raise ValueError("segmentation must have shape [height, width, channels]")
    target_ids = (
        (int(target_geom_id),)
        if isinstance(target_geom_id, int)
        else tuple(int(value) for value in target_geom_id)
    )
    if not target_ids:
        raise ValueError("at least one target geometry id is required")
    mask = np.isin(segmentation[..., 0], target_ids)
    ys, xs = np.nonzero(mask)
    if xs.size < min_visible_pixels:
        return DoorObservationLabel(0.0, 0.0, 0.0, 0.0)
    height, width = mask.shape
    center_x = (float(xs.min()) + float(xs.max()) + 1.0) / (2.0 * width)
    center_y = (float(ys.min()) + float(ys.max()) + 1.0) / (2.0 * height)
    scale = float(np.sqrt(xs.size / float(width * height)))
    return DoorObservationLabel(1.0, center_x, center_y, scale)


def decode_observability(raw: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(raw)


def observability_loss(raw: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    visibility_loss = nn.functional.binary_cross_entropy_with_logits(
        raw[:, 0],
        labels[:, 0],
    )
    positive = labels[:, 0] > 0.5
    if not torch.any(positive):
        return visibility_loss
    regression = nn.functional.smooth_l1_loss(
        torch.sigmoid(raw[positive, 1:]),
        labels[positive, 1:],
    )
    return visibility_loss + 2.0 * regression


def observability_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    visibility_threshold: float = 0.5,
) -> DoorObservabilityMetrics:
    predicted = np.asarray(predictions, dtype=np.float64)
    expected = np.asarray(labels, dtype=np.float64)
    if predicted.shape != expected.shape or predicted.ndim != 2:
        raise ValueError("predictions and labels must have the same [samples, outputs] shape")
    if predicted.shape[1] != 4:
        raise ValueError("observability arrays must have four outputs")
    positive = expected[:, 0] > 0.5
    negative = ~positive
    decisions = predicted[:, 0] >= visibility_threshold
    centroid_error = None
    if np.any(positive):
        delta = predicted[positive, 1:3] - expected[positive, 1:3]
        errors = np.sqrt(delta[:, 0] ** 2 + (0.75 * delta[:, 1]) ** 2)
        centroid_error = float(np.median(errors))
    return DoorObservabilityMetrics(
        sample_count=int(expected.shape[0]),
        positive_count=int(np.sum(positive)),
        negative_count=int(np.sum(negative)),
        visibility_auroc=_binary_auroc(predicted[:, 0], positive),
        visibility_recall=(
            float(np.mean(decisions[positive])) if np.any(positive) else None
        ),
        false_positive_rate=(
            float(np.mean(decisions[negative])) if np.any(negative) else None
        ),
        centroid_median_error_widths=centroid_error,
    )


def _binary_auroc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    positive_scores = scores[positive]
    negative_scores = scores[~positive]
    if positive_scores.size == 0 or negative_scores.size == 0:
        return None
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return float(np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0))
