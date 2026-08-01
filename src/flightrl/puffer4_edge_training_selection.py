from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import torch

from flightrl.puffer4_edge_schema import EDGE_FRAME_PIXELS, EDGE_OBSERVATION_DIM


VISUAL_DEPENDENCE_ABSOLUTE_MARGIN = 1.0e-4
VISUAL_DEPENDENCE_RELATIVE_MARGIN = 0.05


def cyclic_selection_frame_ablation(observation: torch.Tensor) -> torch.Tensor:
    """Break image-agent alignment while preserving every frame and other input."""
    if (
        not isinstance(observation, torch.Tensor)
        or observation.ndim != 2
        or observation.shape[1] != EDGE_OBSERVATION_DIM
    ):
        raise ValueError("edge selection observation shape is incompatible")
    if observation.shape[0] < 2:
        raise ValueError("visual ablation requires at least two selection agents")
    result = observation.clone()
    result[:, :EDGE_FRAME_PIXELS] = observation[:, :EDGE_FRAME_PIXELS].roll(1, dims=0)
    return result


def visual_dependence_margin(clean_decision_action_loss: float) -> float:
    clean = _finite_nonnegative(clean_decision_action_loss, "clean action loss")
    return max(
        VISUAL_DEPENDENCE_ABSOLUTE_MARGIN,
        VISUAL_DEPENDENCE_RELATIVE_MARGIN * clean,
    )


def visual_dependence_check(
    clean_decision_action_loss: float,
    ablated_decision_action_loss: float,
) -> bool:
    clean = _finite_nonnegative(clean_decision_action_loss, "clean action loss")
    ablated = _finite_nonnegative(
        ablated_decision_action_loss,
        "ablated action loss",
    )
    return ablated - clean >= visual_dependence_margin(clean)


def edge_baseline_checks(
    clean_metrics: Mapping[str, float],
    ablated_metrics: Mapping[str, float],
    baselines: Mapping[str, Mapping[str, float]],
) -> dict[str, bool]:
    return {
        "previous_action": clean_metrics["decision_action_loss"]
        < baselines["previous_action"]["decision_action_loss"],
        "constant_grounding": clean_metrics["grounding_loss"]
        < baselines["constant_grounding"]["grounding_loss"],
        "visual_dependence": visual_dependence_check(
            clean_metrics["decision_action_loss"],
            ablated_metrics["decision_action_loss"],
        ),
    }


def _finite_nonnegative(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"edge visual dependence {label} is invalid")
    return float(value)
