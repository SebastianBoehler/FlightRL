from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from flightrl.puffer4_door_bundle import (
    FixedDoorCheckpointBundle,
    load_fixed_door_checkpoint_bundle,
)
from flightrl.puffer4_door_contract import CORRECTED_DOOR_ACTION_CONTRACT
from flightrl.puffer4_door_evidence_age_contract import (
    approved_door_evidence_age_contract_from_report,
)
from flightrl.puffer4_door_stream_contract import verify_door_stream_contract


@dataclass(frozen=True, slots=True)
class PromotionSelectionInput:
    bundle: FixedDoorCheckpointBundle
    report: Mapping[str, Any]
    environment: Mapping[str, Any]
    native_fingerprint: Mapping[str, Any]
    stream_contract: Mapping[str, Any]
    evidence_age_contract: Mapping[str, Any]
    full: Mapping[str, Any]
    masked: Mapping[str, Any]
    recurrence: Mapping[str, Any]
    temporal: Mapping[str, Any]
    live_cap: Mapping[str, Any]
    parent: Mapping[str, str]

    def identity(self) -> dict[str, Any]:
        return {
            "checkpoint": _path_identity(self.bundle.checkpoint_path),
            "promotion_report": _path_identity(self.bundle.report_path),
            "lineage_report": {
                "path": str(self.bundle.lineage_report_path),
                "sha256": self.bundle.lineage_report_sha256,
            },
            "parent_checkpoint": dict(self.parent),
        }


@dataclass(frozen=True, slots=True)
class ScreenSelectionInput:
    seed: int
    bundle: FixedDoorCheckpointBundle
    report: Mapping[str, Any]
    parent: Mapping[str, str]
    budget: Mapping[str, Any]
    full: Mapping[str, Any]
    masked: Mapping[str, Any]

    def identity(self) -> dict[str, Any]:
        return {
            "report": _path_identity(self.bundle.report_path),
            "checkpoint": _path_identity(self.bundle.checkpoint_path),
            "parent_checkpoint": dict(self.parent),
        }


def load_promotion_selection_input(
    checkpoint: str | Path,
    report_path: str | Path,
) -> PromotionSelectionInput:
    bundle = load_fixed_door_checkpoint_bundle(checkpoint, report_path)
    report = bundle.raw_report
    if report.get("evaluation_schema") != "flightrl.fixed_door.promotion.v3":
        raise ValueError("selection requires a canonical v3 promotion report")
    identity = _mapping(report.get("evaluation_identity"), "evaluation identity")
    environment = _mapping(identity.get("environment"), "evaluation environment")
    trained = bundle.trained_identity()
    if identity.get("action_contract_sha256") != bundle.action_contract.sha256():
        raise ValueError("evaluation action contract does not match checkpoint")
    if identity.get("policy_contract_sha256") != bundle.policy_contract["sha256"]:
        raise ValueError("evaluation policy contract does not match checkpoint")
    if environment.get("name") != bundle.env_name:
        raise ValueError("evaluation environment does not match checkpoint")
    native = _mapping(
        identity.get("native_build_fingerprint"),
        "native build fingerprint",
    )
    stream = _mapping(
        identity.get("procedural_stream_contract"),
        "evaluation stream contract",
    )
    verify_door_stream_contract(stream)
    evidence_age = _mapping(
        identity.get("evidence_age_runtime_contract"),
        "evidence-age contract",
    )
    approved_door_evidence_age_contract_from_report(evidence_age)
    steps = _positive_int(environment.get("steps_per_condition"), "steps")
    agents = _positive_int(environment.get("agents"), "agents")
    _positive_int(environment.get("seed"), "evaluation seed", allow_zero=True)
    full = _complete_run(report.get("full_camera"), "full", steps, agents)
    masked = _complete_run(report.get("masked_camera"), "masked", steps, agents)
    recurrence = _complete_nested_run(
        report.get("recurrence_reset_ablation"),
        "recurrence",
        steps,
        agents,
    )
    temporal = _complete_nested_run(
        report.get("temporal_order_ablation"),
        "temporal",
        steps,
        agents,
    )
    live_cap = _complete_nested_run(
        report.get("live_yaw_cap_challenge"),
        "live-cap",
        steps,
        agents,
    )
    lineage = load_fixed_door_checkpoint_bundle(
        bundle.checkpoint_path,
        bundle.lineage_report_path,
    )
    if lineage.trained_identity() != trained:
        raise ValueError("promotion checkpoint lineage does not match")
    return PromotionSelectionInput(
        bundle=bundle,
        report=report,
        environment=environment,
        native_fingerprint=native,
        stream_contract=stream,
        evidence_age_contract=evidence_age,
        full=full,
        masked=masked,
        recurrence=recurrence,
        temporal=temporal,
        live_cap=live_cap,
        parent=_parent_identity(lineage.raw_report),
    )


def load_screen_selection_input(
    report_path: str | Path,
    *,
    seed: int,
) -> ScreenSelectionInput:
    path = Path(report_path).resolve()
    report = _read_object(path)
    checkpoint_value = report.get("checkpoint")
    if not isinstance(checkpoint_value, str):
        raise ValueError("screen report has no checkpoint path")
    checkpoint = Path(checkpoint_value).resolve()
    if path != checkpoint.with_suffix(".report.json"):
        raise ValueError("screen report path is not canonical for its checkpoint")
    bundle = load_fixed_door_checkpoint_bundle(checkpoint, path)
    if bundle.action_contract != CORRECTED_DOOR_ACTION_CONTRACT:
        raise ValueError("screen report is not corrected-yaw BC")
    if bundle.stream_contract is None:
        raise ValueError("screen report has no procedural stream contract")
    config = _mapping(report.get("config"), "screen config")
    if (
        config.get("seed") != seed
        or config.get("evaluation_seed") != 10_000 + seed
        or config.get("bootstrap_max_policy_rollin") != 0.0
        or config.get("rollouts") != 0
        or config.get("fresh_control") is not True
        or report.get("selected_stage") != "bootstrap"
    ):
        raise ValueError(f"screen seed {seed} is not matched corrected-yaw BC")
    evaluation = _mapping(report.get("evaluation"), "screen evaluation")
    if evaluation.get("evaluation_mode") != "full_camera_and_masked_camera":
        raise ValueError("screen evaluation mode is incomplete")
    full = _screen_metrics(
        evaluation.get("full_camera"),
        "full",
        ("success_rate", "collision_rate"),
    )
    masked = _screen_metrics(
        evaluation.get("masked_camera"),
        "masked",
        ("success_rate",),
    )
    excluded = {
        "seed",
        "evaluation_seed",
        "output_dir",
        "puffer_root",
        "source_checkpoint",
        "source_report",
        "skip_build",
    }
    budget = {key: value for key, value in config.items() if key not in excluded}
    return ScreenSelectionInput(
        seed=seed,
        bundle=bundle,
        report=report,
        parent=_parent_identity(report),
        budget=budget,
        full=full,
        masked=masked,
    )


def _complete_nested_run(
    value: object,
    label: str,
    steps: int,
    agents: int,
) -> Mapping[str, Any]:
    outer = _mapping(value, f"{label} evidence")
    return _complete_run(outer.get("metrics"), label, steps, agents)


def _complete_run(
    value: object,
    label: str,
    steps: int,
    agents: int,
) -> Mapping[str, Any]:
    run = _mapping(value, f"{label} metrics")
    finite = _mapping(run.get("finite_outputs"), f"{label} finite outputs")
    performance = _mapping(run.get("performance"), f"{label} performance")
    if (
        run.get("status") != "complete"
        or finite.get("passed") is not True
        or run.get("requested_steps") != steps
        or run.get("completed_steps") != steps
        or performance.get("batch_agents") != agents
    ):
        raise ValueError(f"{label} evaluation is incomplete or non-finite")
    for key in ("success_rate", "outside_fov_success_rate", "collision_rate"):
        _rate(run.get(key), f"{label} {key}")
    return run


def _screen_metrics(
    value: object,
    label: str,
    keys: tuple[str, ...],
) -> Mapping[str, Any]:
    metrics = _mapping(value, f"screen {label} metrics")
    for key in keys:
        _rate(metrics.get(key), f"screen {label} {key}")
    return metrics


def _parent_identity(report: Mapping[str, Any]) -> dict[str, str]:
    path_value = report.get("source_checkpoint")
    digest = report.get("source_checkpoint_sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise ValueError("training report has no exact parent checkpoint lineage")
    path = Path(path_value).resolve()
    if _file_sha256(path) != digest:
        raise ValueError("parent checkpoint SHA-256 does not match lineage")
    config = report.get("config")
    if isinstance(config, Mapping) and isinstance(
        config.get("source_checkpoint"), str
    ):
        if Path(config["source_checkpoint"]).resolve() != path:
            raise ValueError("configured parent checkpoint does not match lineage")
    return {"path": str(path), "sha256": digest}


def _path_identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _file_sha256(path)}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing or invalid")
    return value


def _rate(value: object, label: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return parsed


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    parsed = int(value)
    if parsed != value or parsed < (0 if allow_zero else 1):
        raise ValueError(f"{label} must be an integer")
    return parsed


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("selection input must contain a JSON object")
    return value

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
