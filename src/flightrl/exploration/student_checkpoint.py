from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from math import isfinite
from pathlib import Path

import torch

from flightrl.puffer4_edge_training_state import edge_state_dict_sha256

from .policy import CoverageExplorationActor
from .student_metrics import COVERAGE_CAUSAL_CHECK_NAMES, coverage_causal_checks
from .student_sequence import coverage_sequence_metadata
from .student_training import (
    COVERAGE_TRAINING_REPORT_SCHEMA,
    CoverageTrainConfig,
)


COVERAGE_CHECKPOINT_SCHEMA = "flightrl.coverage.student_checkpoint.v1"
_REPORT_FIELDS = {
    "schema",
    "status",
    "config",
    "datasets",
    "selection",
    "selection_history_permuted",
    "persistence_baseline",
    "telemetry_only_baseline",
    "causal_gate",
    "selected_actor_state_sha256",
    "parameter_count",
    "evaluation_scope",
    "closed_loop_evaluated",
    "generalization_authority",
    "training_authority",
    "deployment_authority",
    "flight_authority",
}
_METRIC_FIELDS = {
    "action_loss",
    "decision_action_loss",
    "decision_mode_accuracy",
    "decision_samples",
    "matched_pair_action_loss",
    "matched_pair_mode_accuracy",
    "matched_pair_samples",
}


def save_coverage_checkpoint(
    path: str | Path,
    actor: CoverageExplorationActor,
    report: dict,
) -> Path:
    _require_passed_actor_report(actor, report)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": COVERAGE_CHECKPOINT_SCHEMA,
        "actor": "CoverageExplorationActor",
        "hidden_size": actor.hidden_size,
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in actor.state_dict().items()
        },
        "training_report": report,
        "training_authority": False,
        "deployment_authority": False,
        "flight_authority": False,
    }
    torch.save(payload, output)
    return output


def load_coverage_checkpoint(
    path: str | Path,
) -> tuple[CoverageExplorationActor, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    expected = {
        "schema",
        "actor",
        "hidden_size",
        "state_dict",
        "training_report",
        "training_authority",
        "deployment_authority",
        "flight_authority",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("coverage checkpoint fields are incompatible")
    if (
        payload["schema"] != COVERAGE_CHECKPOINT_SCHEMA
        or payload["actor"] != "CoverageExplorationActor"
        or any(
            payload[name] is not False
            for name in (
                "training_authority",
                "deployment_authority",
                "flight_authority",
            )
        )
    ):
        raise ValueError("coverage checkpoint contract is incompatible")
    state = payload["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("coverage checkpoint state is incompatible")
    actor = CoverageExplorationActor(hidden_size=payload["hidden_size"])
    actor.load_state_dict(state, strict=True)
    actor.eval()
    report = payload["training_report"]
    _require_passed_actor_report(actor, report)
    return actor, report


def _require_passed_actor_report(
    actor: CoverageExplorationActor, report: object
) -> None:
    if type(actor) is not CoverageExplorationActor:
        raise TypeError("coverage checkpoint actor type is incompatible")
    if (
        not isinstance(report, dict)
        or report.get("schema") != COVERAGE_TRAINING_REPORT_SCHEMA
        or report.get("status") != "complete"
        or not isinstance(report.get("causal_gate"), dict)
        or report["causal_gate"].get("passed") is not True
    ):
        raise ValueError("coverage checkpoint requires a passed causal gate")
    if (
        set(report) != _REPORT_FIELDS
        or report.get("evaluation_scope") != "offline_sequence_behavior_cloning"
        or report.get("closed_loop_evaluated") is not False
        or type(report.get("parameter_count")) is not int
        or report["parameter_count"] != actor.parameter_count
    ):
        raise ValueError("coverage checkpoint report contract is incompatible")
    _require_training_config(report.get("config"))
    if set(report["causal_gate"]) != {"checks", "passed"}:
        raise ValueError("coverage checkpoint report contract is incompatible")
    checks = report["causal_gate"].get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != set(COVERAGE_CAUSAL_CHECK_NAMES)
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("coverage checkpoint causal check details are incompatible")
    metric_names = (
        "selection",
        "selection_history_permuted",
        "persistence_baseline",
        "telemetry_only_baseline",
    )
    if any(not isinstance(report.get(name), dict) for name in metric_names):
        raise ValueError("coverage checkpoint causal metrics are incompatible")
    for name in metric_names:
        _require_metric_record(report[name])
    try:
        derived_checks = coverage_causal_checks(
            report["selection"],
            report["selection_history_permuted"],
            report["persistence_baseline"],
            report["telemetry_only_baseline"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "coverage checkpoint causal metrics are incompatible"
        ) from error
    if derived_checks != checks:
        raise ValueError(
            "coverage checkpoint metrics do not support its causal checks"
        )
    _require_dataset_digests(report.get("datasets"))
    if report.get("selected_actor_state_sha256") != edge_state_dict_sha256(
        actor.state_dict()
    ):
        raise ValueError("coverage checkpoint report does not bind the actor state")
    if any(
        report.get(name) is not False
        for name in (
            "generalization_authority",
            "training_authority",
            "deployment_authority",
            "flight_authority",
        )
    ):
        raise ValueError("coverage checkpoint report grants unsupported authority")


def _require_dataset_digests(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"train", "selection"}:
        raise ValueError("coverage checkpoint dataset provenance is incompatible")
    scene_sets = []
    for split, record in value.items():
        if not isinstance(record, dict) or record.get("split") != split:
            raise ValueError("coverage checkpoint dataset provenance is incompatible")
        metadata = {name: field for name, field in record.items() if name != "sha256"}
        scene_ids = metadata.get("scene_ids")
        if not isinstance(scene_ids, list):
            raise ValueError("coverage checkpoint dataset provenance is incompatible")
        try:
            expected = coverage_sequence_metadata(
                split=split,
                steps=metadata.get("steps"),
                scene_ids=tuple(scene_ids),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "coverage checkpoint dataset provenance is incompatible"
            ) from error
        if metadata != expected:
            raise ValueError("coverage checkpoint dataset provenance is incompatible")
        scene_sets.append(set(scene_ids))
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("coverage checkpoint dataset digest is incompatible")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(
                "coverage checkpoint dataset digest is incompatible"
            ) from error
    if scene_sets[0] & scene_sets[1]:
        raise ValueError("coverage checkpoint dataset splits overlap")


def _require_training_config(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("coverage checkpoint training config is incompatible")
    try:
        parsed = CoverageTrainConfig(**value)
    except (TypeError, ValueError) as error:
        raise ValueError("coverage checkpoint training config is incompatible") from error
    if asdict(parsed) != value:
        raise ValueError("coverage checkpoint training config is incompatible")


def _require_metric_record(value: dict[str, object]) -> None:
    if set(value) != _METRIC_FIELDS:
        raise ValueError("coverage checkpoint causal metrics are incompatible")
    for name, field in value.items():
        if name.endswith("samples"):
            valid = type(field) is int and field > 0
        else:
            valid = (
                not isinstance(field, bool)
                and isinstance(field, (int, float))
                and isfinite(float(field))
                and float(field) >= 0.0
            )
            if name.endswith("accuracy"):
                valid = valid and float(field) <= 1.0
        if not valid:
            raise ValueError("coverage checkpoint causal metrics are incompatible")
