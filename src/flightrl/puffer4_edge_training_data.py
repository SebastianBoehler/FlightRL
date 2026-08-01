from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from flightrl.puffer4_edge_sequence import (
    EdgeSequenceDataset,
    require_edge_sequence_structure,
)


CRITICAL_DECISION_BOOST = 8.0
MATERIAL_ACTION_SWITCH = 0.05


@dataclass(frozen=True, slots=True)
class EdgeLossWeights:
    episode: torch.Tensor
    decision: torch.Tensor
    visibility: torch.Tensor
    box: torch.Tensor
    critical: torch.Tensor


def balanced_visibility_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return (
        F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights
    ).mean()


def empty_loss_totals() -> dict[str, float]:
    return {
        name: 0.0
        for name in (
            "total",
            "decision_action",
            "frame_action",
            "visibility",
            "box",
            "grounding",
        )
    }


def accumulate_losses(
    totals: dict[str, float],
    losses: dict[str, torch.Tensor],
) -> None:
    for name, value in losses.items():
        totals[name] += float(value.detach())


def average_losses(totals: dict[str, float], steps: int) -> dict[str, float]:
    metrics = {f"{name}_loss": value / steps for name, value in totals.items()}
    metrics["selection_score"] = metrics["decision_action_loss"]
    return metrics


def edge_sequence_loss_weights(dataset: EdgeSequenceDataset) -> EdgeLossWeights:
    require_edge_sequence_structure(dataset)
    episode = _episode_balanced_weights(dataset.resets)
    critical = _critical_decisions(dataset)
    decision = _normalized(episode * np.where(critical, CRITICAL_DECISION_BOOST, 1.0))
    visible = dataset.grounding[..., 0] > 0.5
    visibility = _class_balanced_weights(episode, visible)
    box = _visible_weights(episode, visible)
    return EdgeLossWeights(
        episode=torch.from_numpy(episode),
        decision=torch.from_numpy(decision),
        visibility=torch.from_numpy(visibility),
        box=torch.from_numpy(box),
        critical=torch.from_numpy(critical),
    )


def edge_training_baseline_values(
    dataset: EdgeSequenceDataset,
    *,
    visibility_loss_weight: float,
    box_loss_weight: float,
) -> dict[str, dict[str, float | list[float]]]:
    weights = edge_sequence_loss_weights(dataset)
    target_action = torch.from_numpy(dataset.teacher_actions)
    previous_action = torch.from_numpy(dataset.telemetry[..., 15:19])
    action_error = (
        (previous_action - target_action).square()
        * torch.tensor((1.0, 0.25, 0.25, 1.0))
    ).mean(dim=-1)
    target_grounding = torch.from_numpy(dataset.grounding)
    visible = target_grounding[..., 0]
    probability = float((weights.visibility * visible).mean())
    probability = min(max(probability, 1.0e-5), 1.0 - 1.0e-5)
    visibility_loss = float(
        (
            weights.visibility
            * -(
                visible * np.log(probability)
                + (1.0 - visible) * np.log(1.0 - probability)
            )
        ).mean()
    )
    if bool((weights.box > 0.0).any()):
        box = (weights.box.unsqueeze(-1) * target_grounding[..., 1:]).sum(
            (0, 1)
        ) / weights.box.sum()
        difference = (target_grounding[..., 1:] - box).abs()
        smooth_l1 = torch.where(
            difference < 1.0,
            0.5 * difference.square(),
            difference - 0.5,
        ).mean(dim=-1)
        box_loss = float((weights.box * smooth_l1).mean())
    else:
        box = torch.zeros(3)
        box_loss = 0.0
    return {
        "previous_action": {
            "frame_action_loss": float(action_error.mean()),
            "decision_action_loss": float((weights.decision * action_error).mean()),
        },
        "constant_grounding": {
            "visible_probability": probability,
            "box": [float(value) for value in box],
            "visibility_loss": visibility_loss,
            "box_loss": box_loss,
            "grounding_loss": (
                visibility_loss_weight * visibility_loss + box_loss_weight * box_loss
            ),
        },
    }


def edge_step_losses(
    action,
    grounding,
    visibility_logit,
    dataset,
    weights,
    step,
    config,
) -> dict[str, torch.Tensor]:
    device = action.device
    target_action = torch.from_numpy(dataset.teacher_actions[step]).to(device)
    target_grounding = torch.from_numpy(dataset.grounding[step]).to(device)
    action_weights = torch.tensor((1.0, 0.25, 0.25, 1.0), device=device)
    action_error = ((action - target_action).square() * action_weights).mean(-1)
    decision_action = (weights.decision[step].to(device) * action_error).mean()
    visible = target_grounding[:, 0]
    visibility_loss = balanced_visibility_loss(
        visibility_logit,
        visible,
        weights.visibility[step].to(device),
    )
    box_error = F.smooth_l1_loss(
        grounding[:, 1:], target_grounding[:, 1:], reduction="none"
    ).mean(-1)
    box_loss = (weights.box[step].to(device) * box_error).mean()
    grounding_loss = (
        config.visibility_loss_weight * visibility_loss
        + config.box_loss_weight * box_loss
    )
    return {
        "total": decision_action + grounding_loss,
        "decision_action": decision_action,
        "frame_action": action_error.mean(),
        "visibility": visibility_loss,
        "box": box_loss,
        "grounding": grounding_loss,
    }


def _episode_balanced_weights(resets: np.ndarray) -> np.ndarray:
    steps, agents = resets.shape
    result = np.empty((steps, agents), dtype=np.float32)
    for agent in range(agents):
        starts = np.flatnonzero(resets[:, agent])
        ends = np.r_[starts[1:], steps]
        for start, end in zip(starts, ends, strict=True):
            result[start:end, agent] = 1.0 / (end - start)
    return _normalized(result)


def _critical_decisions(dataset: EdgeSequenceDataset) -> np.ndarray:
    reset = dataset.resets.astype(bool)
    visible = dataset.grounding[..., 0] > 0.5
    critical = reset.copy()
    if dataset.shape[0] > 1:
        continuation = ~reset[1:]
        critical[1:] |= continuation & (visible[1:] != visible[:-1])
        action_delta = np.max(
            np.abs(dataset.teacher_actions[1:] - dataset.teacher_actions[:-1]),
            axis=-1,
        )
        critical[1:] |= continuation & (action_delta >= MATERIAL_ACTION_SWITCH)
    return critical


def _class_balanced_weights(base: np.ndarray, positive: np.ndarray) -> np.ndarray:
    positive_mass = float(base[positive].sum())
    negative_mass = float(base[~positive].sum())
    if positive_mass == 0.0 or negative_mass == 0.0:
        return base.copy()
    half = 0.5 * float(base.sum())
    result = base.copy()
    result[positive] *= half / positive_mass
    result[~positive] *= half / negative_mass
    return _normalized(result)


def _visible_weights(base: np.ndarray, visible: np.ndarray) -> np.ndarray:
    result = np.where(visible, base, 0.0)
    return _normalized(result) if bool(visible.any()) else result.astype(np.float32)


def _normalized(values: np.ndarray) -> np.ndarray:
    return (values / float(values.mean())).astype(np.float32, copy=False)
