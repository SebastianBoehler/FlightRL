from __future__ import annotations

import torch


def calibrate_visibility_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    min_recall: float = 0.95,
) -> float:
    probabilities = torch.sigmoid(logits.detach()[:, 0])
    positive_scores = torch.sort(probabilities[labels[:, 0] > 0.5]).values
    if positive_scores.numel() == 0:
        return 0.5
    allowed_misses = int((1.0 - min_recall) * positive_scores.numel())
    index = min(allowed_misses, positive_scores.numel() - 1)
    return float(positive_scores[index])


def grounding_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    visibility_threshold: float = 0.5,
) -> dict[str, float]:
    probabilities = torch.sigmoid(logits.detach()[:, :2])
    positive = labels[:, 0] > 0.5
    negative = ~positive
    decisions = probabilities[:, 0] >= visibility_threshold
    centroid_error = torch.abs(probabilities[positive, 1] - labels[positive, 1])
    ranks = torch.empty_like(probabilities[:, 0])
    order = torch.argsort(probabilities[:, 0])
    ranks[order] = torch.arange(
        1,
        probabilities.shape[0] + 1,
        dtype=probabilities.dtype,
    )
    positive_count = torch.sum(positive)
    negative_count = torch.sum(negative)
    auroc = (
        (
            torch.sum(ranks[positive])
            - positive_count * (positive_count + 1) / 2
        )
        / (positive_count * negative_count)
        if positive_count > 0 and negative_count > 0
        else probabilities.new_tensor(float("nan"))
    )
    return {
        "visibility_auroc": float(auroc),
        "visibility_recall": (
            float(torch.mean(decisions[positive].float())) if torch.any(positive) else 0.0
        ),
        "visibility_false_positive_rate": (
            float(torch.mean(decisions[negative].float())) if torch.any(negative) else 0.0
        ),
        "centroid_median_error_widths": (
            float(torch.median(centroid_error)) if torch.any(positive) else 1.0
        ),
        "positive_fraction": float(torch.mean(positive.float())),
        "visibility_threshold": visibility_threshold,
    }


def grounding_selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics.get("visibility_recall", 0.0)
        - 2.0 * metrics.get("visibility_false_positive_rate", 1.0)
        - metrics.get("centroid_median_error_widths", 1.0)
    )


def fixed_door_grounder_gate(metrics: dict[str, float]) -> dict:
    checks = {
        "visibility_auroc": metrics.get("visibility_auroc", 0.0) >= 0.90,
        "centroid": metrics.get("centroid_median_error_widths", 1.0) <= 0.12,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }
