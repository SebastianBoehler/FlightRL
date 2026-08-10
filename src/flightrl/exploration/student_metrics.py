from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import numpy as np
import torch

from flightrl.puffer4_edge_schema import EDGE_FRAME_PIXELS, TELEMETRY_SPECS

from .contract import COVERAGE_MAXIMUM_YAW_RATE_DEG_S
from .student_sequence import (
    CoverageSequenceDataset,
    EVENT_ENTER_SCAN,
    EVENT_RESUME_ADVANCE,
    require_coverage_sequence_dataset,
)


VISUAL_DEPENDENCE_ABSOLUTE_MARGIN = 1.0e-4
VISUAL_DEPENDENCE_RELATIVE_MARGIN = 0.05
COVERAGE_CAUSAL_CHECK_NAMES = (
    "persistence",
    "telemetry_only",
    "entire_image_history_permutation",
    "decision_mode_camera_dependence",
    "matched_counterfactual",
)
_MODE_ACTIONS = np.asarray(((0.5, 0.0), (0.0, 1.0)), dtype=np.float32)


def decision_event_mask(dataset: CoverageSequenceDataset) -> np.ndarray:
    """Select teacher mode transitions and explicit matched counterfactuals."""
    require_coverage_sequence_dataset(dataset)
    transition = np.isin(
        dataset.event_labels,
        np.asarray((EVENT_ENTER_SCAN, EVENT_RESUME_ADVANCE), dtype=np.uint8),
    )
    mask = transition | (dataset.pair_ids >= 0)
    if not np.any(mask):
        raise ValueError("coverage selection has no decision events")
    return mask


def history_permuted_observation(
    dataset: CoverageSequenceDataset, step: int
) -> torch.Tensor:
    """Use one cyclic agent reassignment for every frame in the sequence."""
    if dataset.shape[1] < 2:
        raise ValueError("coverage history permutation requires at least two agents")
    observation = dataset.model_observation(step)
    result = observation.clone()
    result[:, :EDGE_FRAME_PIXELS] = observation[:, :EDGE_FRAME_PIXELS].roll(
        1, dims=0
    )
    return result


def persistence_baseline_metrics(
    dataset: CoverageSequenceDataset,
) -> dict[str, float | int]:
    require_coverage_sequence_dataset(dataset)
    prediction = np.stack(
        (
            dataset.telemetry[..., 15],
            dataset.telemetry[..., 18]
            * (TELEMETRY_SPECS[18][2] / COVERAGE_MAXIMUM_YAW_RATE_DEG_S),
        ),
        axis=-1,
    ).clip(-1.0, 1.0)
    return coverage_action_metrics(prediction, dataset)


def coverage_action_metrics(
    prediction: np.ndarray, dataset: CoverageSequenceDataset
) -> dict[str, float | int]:
    expected = dataset.teacher_actions.shape
    values = np.asarray(prediction, dtype=np.float32)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError("coverage action predictions are incompatible")
    squared = np.square(values - dataset.teacher_actions)
    decision = decision_event_mask(dataset)
    predicted_mode = _nearest_mode(values[decision])
    teacher_mode = _nearest_mode(dataset.teacher_actions[decision])
    metrics: dict[str, float | int] = {
        "action_loss": float(np.mean(squared, dtype=np.float64)),
        "decision_action_loss": float(np.mean(squared[decision], dtype=np.float64)),
        "decision_mode_accuracy": float(np.mean(predicted_mode == teacher_mode)),
        "decision_samples": int(np.count_nonzero(decision)),
    }
    matched = dataset.pair_ids >= 0
    if np.any(matched):
        metrics.update(
            {
                "matched_pair_action_loss": float(
                    np.mean(squared[matched], dtype=np.float64)
                ),
                "matched_pair_mode_accuracy": float(
                    np.mean(
                        _nearest_mode(values[matched])
                        == _nearest_mode(dataset.teacher_actions[matched])
                    )
                ),
                "matched_pair_samples": int(np.count_nonzero(matched)),
            }
        )
    return metrics


def coverage_causal_checks(
    clean: Mapping[str, float | int],
    permuted: Mapping[str, float | int],
    persistence: Mapping[str, float | int],
    telemetry_only: Mapping[str, float | int],
) -> dict[str, bool]:
    clean_loss = _metric(clean, "decision_action_loss")
    permuted_loss = _metric(permuted, "decision_action_loss")
    clean_accuracy = _metric(clean, "decision_mode_accuracy")
    permuted_accuracy = _metric(permuted, "decision_mode_accuracy")
    clean_pair_accuracy = _metric(clean, "matched_pair_mode_accuracy")
    permuted_pair_accuracy = _metric(permuted, "matched_pair_mode_accuracy")
    return {
        "persistence": clean_loss < _metric(persistence, "decision_action_loss"),
        "telemetry_only": clean_loss
        < _metric(telemetry_only, "decision_action_loss"),
        "entire_image_history_permutation": permuted_loss - clean_loss
        >= max(
            VISUAL_DEPENDENCE_ABSOLUTE_MARGIN,
            VISUAL_DEPENDENCE_RELATIVE_MARGIN * clean_loss,
        ),
        "decision_mode_camera_dependence": clean_accuracy >= 0.80
        and clean_accuracy - permuted_accuracy >= 0.25,
        "matched_counterfactual": clean_pair_accuracy >= 0.80
        and clean_pair_accuracy - permuted_pair_accuracy >= 0.25,
    }


def _nearest_mode(actions: np.ndarray) -> np.ndarray:
    distances = np.square(actions[..., None, :] - _MODE_ACTIONS).sum(axis=-1)
    return np.argmin(distances, axis=-1)


def _metric(metrics: Mapping[str, float | int], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"coverage metric {name} is missing")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"coverage metric {name} is invalid")
    return result
