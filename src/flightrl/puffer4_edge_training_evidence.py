from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isclose, isfinite

from flightrl.puffer4_edge_perception_warmup import (
    WARMUP_REPORT_FIELDS,
    edge_perception_parameter_names,
    edge_perception_state_sha256,
    evaluate_edge_grounding,
)
from flightrl.puffer4_edge_training import EdgeTrainConfig, edge_training_baselines
from flightrl.puffer4_edge_training_selection import edge_baseline_checks


LOSS_FIELDS = {
    "total_loss",
    "decision_action_loss",
    "frame_action_loss",
    "visibility_loss",
    "box_loss",
    "grounding_loss",
    "selection_score",
}
HISTORY_FIELDS = {
    "epoch",
    "train",
    "selection",
    "selection_visual_ablation",
    "baseline_checks",
}
WARMUP_METRIC_FIELDS = {
    "visibility_loss",
    "box_loss",
    "grounding_loss",
}


def require_perception_warmup_evidence(
    value: object,
    config: EdgeTrainConfig,
    actor,
    train,
    selection,
) -> None:
    if not isinstance(value, Mapping) or set(value) != WARMUP_REPORT_FIELDS:
        raise ValueError("edge perception warmup fields are incompatible")
    history = value["history"]
    if not isinstance(history, list) or len(history) != config.warmup_epochs:
        raise ValueError("edge perception warmup history is incomplete")
    records = []
    for epoch, record in enumerate(history, 1):
        if (
            not isinstance(record, Mapping)
            or set(record) != {"epoch", "selection"}
            or record["epoch"] != epoch
        ):
            raise ValueError("edge perception warmup epoch is invalid")
        records.append((epoch, _require_warmup_metrics(record["selection"], config)))
    selected_epoch, selected_metrics = min(
        records, key=lambda item: item[1]["grounding_loss"]
    )
    if (
        value["selected_epoch"] != selected_epoch
        or value["selected_selection_metrics"] != selected_metrics
    ):
        raise ValueError("edge perception warmup selection is inconsistent")
    expected_names = list(edge_perception_parameter_names(actor))
    if value["frozen_parameter_names"] != expected_names:
        raise ValueError("edge perception frozen parameter names are incompatible")
    if value["selected_state_sha256"] != edge_perception_state_sha256(actor):
        raise ValueError("edge perception selected state digest does not match")
    expected_sampling = {
        "rng": "numpy.PCG64",
        "flattening": "step_major_agent_minor",
        "order": "full_permutation_without_replacement",
        "samples_per_epoch": train.shape[0] * train.shape[1],
        "batches_per_epoch": (
            train.shape[0] * train.shape[1] // config.warmup_batch_size
        ),
    }
    if value["sampling"] != expected_sampling:
        raise ValueError("edge perception warmup sampling is incompatible")
    if evaluate_edge_grounding(actor, selection, config) != selected_metrics:
        raise ValueError("edge perception selected metrics do not reproduce")


def _require_warmup_metrics(value: object, config) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != WARMUP_METRIC_FIELDS:
        raise ValueError("edge perception warmup metrics are incompatible")
    if any(not _finite_number(item) or float(item) < 0.0 for item in value.values()):
        raise ValueError("edge perception warmup metrics must be finite")
    metrics = {name: float(item) for name, item in value.items()}
    expected = (
        config.visibility_loss_weight * metrics["visibility_loss"]
        + config.box_loss_weight * metrics["box_loss"]
    )
    if not isclose(
        metrics["grounding_loss"], expected, rel_tol=1e-7, abs_tol=1e-8
    ):
        raise ValueError("edge perception warmup metrics are inconsistent")
    return metrics


def require_training_selection_evidence(
    report: Mapping,
    selection,
    config: EdgeTrainConfig,
    actor,
    *,
    reproduce: Callable[[bool], dict[str, float]],
) -> None:
    baselines = edge_training_baselines(selection, config)
    if report["baselines"] != baselines:
        raise ValueError("edge training baselines do not reproduce")
    history = report["history"]
    if not isinstance(history, list) or len(history) != config.epochs:
        raise ValueError("edge training history does not cover every epoch")
    eligible = []
    for epoch, record in enumerate(history, 1):
        if (
            not isinstance(record, Mapping)
            or set(record) != HISTORY_FIELDS
            or record["epoch"] != epoch
        ):
            raise ValueError("edge training history epoch fields are invalid")
        require_loss_metrics(record["train"], config)
        clean = require_loss_metrics(record["selection"], config)
        ablated = require_loss_metrics(record["selection_visual_ablation"], config)
        checks = edge_baseline_checks(clean, ablated, baselines)
        if record["baseline_checks"] != checks:
            raise ValueError("edge training epoch baseline checks are inconsistent")
        if all(checks.values()):
            eligible.append((epoch, clean, ablated, checks))
    if not eligible:
        raise ValueError("edge training report has no baseline-passing epoch")
    selected = min(eligible, key=lambda item: item[1]["selection_score"])
    _require_selected_claims(report, selected)
    _require_selected_reproduction(report, reproduce, baselines)


def require_loss_metrics(value: object, config: EdgeTrainConfig) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != LOSS_FIELDS:
        raise ValueError("edge training metric fields are incompatible")
    if any(not _finite_number(item) or float(item) < 0.0 for item in value.values()):
        raise ValueError("edge report requires finite training metrics")
    metrics = {name: float(item) for name, item in value.items()}
    expected_grounding = (
        config.visibility_loss_weight * metrics["visibility_loss"]
        + config.box_loss_weight * metrics["box_loss"]
    )
    if not all(
        (
            isclose(
                metrics["grounding_loss"],
                expected_grounding,
                rel_tol=1e-7,
                abs_tol=1e-8,
            ),
            isclose(
                metrics["total_loss"],
                metrics["decision_action_loss"] + metrics["grounding_loss"],
                rel_tol=1e-7,
                abs_tol=1e-8,
            ),
            metrics["selection_score"] == metrics["decision_action_loss"],
        )
    ):
        raise ValueError("edge training metrics are mathematically inconsistent")
    return metrics


def _require_selected_claims(report: Mapping, selected: tuple) -> None:
    epoch, clean, ablated, checks = selected
    if (
        report["best_epoch"] != epoch
        or report["best_selection_metrics"] != clean
        or report["best_selection_visual_ablation_metrics"] != ablated
        or report["best_selection_loss"] != clean["selection_score"]
    ):
        raise ValueError("edge training selected epoch is inconsistent")
    expected_gate = {"passed": True, "checks": checks}
    if report["baseline_gate"] != expected_gate:
        raise ValueError("edge training baseline gate is invalid")


def _require_selected_reproduction(
    report: Mapping,
    reproduce: Callable[[bool], dict[str, float]],
    baselines: Mapping,
) -> None:
    clean = reproduce(False)
    ablated = reproduce(True)
    checks = edge_baseline_checks(clean, ablated, baselines)
    if clean != report["best_selection_metrics"]:
        raise ValueError(
            "edge selected clean metrics do not reproduce from actor state"
        )
    if ablated != report["best_selection_visual_ablation_metrics"]:
        raise ValueError(
            "edge selected ablation metrics do not reproduce from actor state"
        )
    if not all(checks.values()) or checks != report["baseline_gate"]["checks"]:
        raise ValueError("edge selected actor baseline checks do not reproduce")


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
    )
