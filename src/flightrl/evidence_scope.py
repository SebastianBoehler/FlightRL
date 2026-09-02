from __future__ import annotations

from pathlib import Path
from typing import Any

from flightrl.artifact_identity import file_identity


DESKTOP_CPU_SCOPE = "desktop_cpu_only"
DESKTOP_DEVELOPMENT_SCOPE = "desktop_development"
EDGE_DEPLOYMENT_SCOPE = "edge_deployment"
EDGE_DEPLOYMENT_VERIFIER_MISSING = "edge_deployment_verifier_missing"


def require_file_identity(
    value: object,
    expected_path: str | Path,
    *,
    label: str,
) -> dict[str, str]:
    expected = file_identity(expected_path)
    if not isinstance(value, dict) or value != expected:
        raise ValueError(f"{label} identity does not match {expected['path']}")
    return expected


def require_existing_file_identity(
    value: object,
    *,
    label: str,
) -> dict[str, str]:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"{label} identity is missing or invalid")
    return require_file_identity(value, value["path"], label=label)


def deployment_claim_failures(report: dict[str, Any]) -> list[str]:
    if report.get("evidence_scope") != EDGE_DEPLOYMENT_SCOPE:
        return ["deployment_scope_invalid"]
    if report.get("deployment_authority") is not True:
        return ["deployment_authority_missing"]
    return []


def deployment_authority_failures(report: dict[str, Any]) -> list[str]:
    failures = deployment_claim_failures(report)
    return failures or [EDGE_DEPLOYMENT_VERIFIER_MISSING]


def has_deployment_authority(report: dict[str, Any]) -> bool:
    return not deployment_authority_failures(report)
