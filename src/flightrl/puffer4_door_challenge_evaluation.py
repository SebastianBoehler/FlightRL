from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)


@dataclass(frozen=True, slots=True)
class MatchedControlReport:
    path: Path
    sha256: str
    metrics: Mapping[str, Any]
    environment: Mapping[str, Any]


def validate_challenge_options(
    *,
    challenge: str | None,
    control_report: Path | None,
    output: Path | None,
    live_yaw_cap_challenge: bool,
) -> None:
    if challenge is None:
        if control_report is not None:
            raise ValueError("--control-report requires --challenge")
        return
    if control_report is None:
        raise ValueError("--challenge requires --control-report")
    if live_yaw_cap_challenge:
        raise ValueError("combined challenge interventions are forbidden")


def resolve_challenge_output(
    checkpoint: Path,
    challenge: str,
    *,
    control_report: Path,
    lineage_report: Path,
    requested: Path | None,
) -> Path:
    checkpoint = checkpoint.resolve()
    canonical = checkpoint.with_suffix(".promotion-evaluation.json").resolve()
    output = (
        checkpoint.with_suffix(f".{challenge}.challenge-evaluation.json")
        if requested is None
        else requested
    ).resolve()
    return _exclusive_output(
        output,
        {
            "canonical evaluation": canonical,
            "control report": control_report.resolve(),
            "lineage report": lineage_report.resolve(),
            "checkpoint": checkpoint,
        },
    )


def resolve_canonical_output(
    checkpoint: Path,
    *,
    lineage_report: Path,
    requested: Path | None,
) -> Path:
    checkpoint = checkpoint.resolve()
    output = (
        checkpoint.with_suffix(".promotion-evaluation.json")
        if requested is None
        else requested
    ).resolve()
    return _exclusive_output(
        output,
        {
            "lineage report": lineage_report.resolve(),
            "checkpoint": checkpoint,
        },
    )


def _exclusive_output(output: Path, aliases: Mapping[str, Path]) -> Path:
    for label, path in aliases.items():
        if output == path:
            raise ValueError(f"evaluation output cannot alias {label}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite challenge output: {output}")
    return output


def load_matched_control_report(
    path: Path,
    *,
    trained_identity: Mapping[str, Any],
    native_build_fingerprint: Mapping[str, Any],
    stream_contract: Mapping[str, Any],
    seed: int,
    steps: int,
    agents: int,
) -> MatchedControlReport:
    resolved = path.resolve()
    report = _read_object(resolved)
    if report.get("evaluation_schema") != "flightrl.fixed_door.promotion.v3":
        raise ValueError("control report is not a canonical v3 promotion report")
    identity = _mapping(report.get("evaluation_identity"), "control identity")
    encoded_report = _mapping(identity.get("report"), "control report identity")
    if (
        identity.get("kind") != "fixed_door_promotion"
        or identity.get("schema_version") != 1
        or Path(str(encoded_report.get("path"))).resolve() != resolved
    ):
        raise ValueError("control report path or identity does not match")
    if report.get("trained_identity") != dict(trained_identity):
        raise ValueError("control report trained identity does not match")
    if identity.get("native_build_fingerprint") != dict(
        native_build_fingerprint
    ):
        raise ValueError("control report native build fingerprint does not match")
    for key in ("action_contract", "policy_contract"):
        if identity.get(f"{key}_sha256") != _contract_sha(
            trained_identity, key
        ):
            raise ValueError(f"control report {key} SHA-256 does not match")
    if identity.get("procedural_stream_contract") != dict(stream_contract):
        raise ValueError("control report stream contract does not match")
    if identity.get("evidence_age_runtime_contract") != (
        FIXED_DOOR_EVIDENCE_AGE_CONTRACT.to_report()
    ):
        raise ValueError("control report evidence-age contract does not match")
    environment = _mapping(
        identity.get("environment"),
        "control evaluation environment",
    )
    expected = {
        "name": _environment_name(trained_identity),
        "seed": seed,
        "steps_per_condition": steps,
        "agents": agents,
    }
    for key, value in expected.items():
        if environment.get(key) != value:
            label = "steps" if key == "steps_per_condition" else key
            raise ValueError(f"control report {label} does not match")
    metrics = _mapping(report.get("full_camera"), "control full-camera metrics")
    finite = _mapping(metrics.get("finite_outputs"), "control finite outputs")
    if metrics.get("status") != "complete" or finite.get("passed") is not True:
        raise ValueError("control full-camera run is incomplete or non-finite")
    return MatchedControlReport(
        path=resolved,
        sha256=_file_sha256(resolved),
        metrics=metrics,
        environment=environment,
    )


def _contract_sha(identity: Mapping[str, Any], key: str) -> str:
    contract = _mapping(identity.get(key), f"trained {key}")
    digest = contract.get("sha256")
    if not isinstance(digest, str):
        raise ValueError(f"trained {key} has no SHA-256")
    return digest


def _environment_name(identity: Mapping[str, Any]) -> str:
    environment = _mapping(identity.get("environment"), "trained environment")
    name = environment.get("name")
    if not isinstance(name, str):
        raise ValueError("trained environment has no name")
    return name


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing or invalid")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("control report must contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
