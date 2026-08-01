from __future__ import annotations

from typing import Any, TypedDict


class AxisAuthority(TypedDict):
    vx_body_m_s: bool
    vy_body_m_s: bool
    vz_m_s: bool
    yawrate_deg_s: bool


class AuthorityDecisionFields(TypedDict):
    candidate_authority: str
    candidate_axis_authority: AxisAuthority
    approved_authority: str
    axis_authority: AxisAuthority


NO_AXIS_AUTHORITY = AxisAuthority(
    vx_body_m_s=False,
    vy_body_m_s=False,
    vz_m_s=False,
    yawrate_deg_s=False,
)
YAW_ONLY_AXIS_AUTHORITY = AxisAuthority(
    vx_body_m_s=False,
    vy_body_m_s=False,
    vz_m_s=False,
    yawrate_deg_s=True,
)
BOUNDED_FORWARD_AXIS_AUTHORITY = AxisAuthority(
    vx_body_m_s=True,
    vy_body_m_s=False,
    vz_m_s=False,
    yawrate_deg_s=True,
)


def yaw_only_authority_fields(passed: bool) -> AuthorityDecisionFields:
    return _authority_fields("yaw_only", YAW_ONLY_AXIS_AUTHORITY, passed)


def bounded_forward_authority_fields(
    passed: bool,
) -> AuthorityDecisionFields:
    return _authority_fields(
        "bounded_forward_yaw",
        BOUNDED_FORWARD_AXIS_AUTHORITY,
        passed,
    )


def authority_decision_errors(
    report: dict[str, Any],
    *,
    candidate_authority: str,
    candidate_axes: AxisAuthority,
) -> list[str]:
    schema_version = report.get("schema_version")
    errors = []
    if schema_version == 2:
        if report.get("candidate_authority") != candidate_authority:
            errors.append("readiness report candidate authority does not match")
        if report.get("candidate_axis_authority") != candidate_axes:
            errors.append("readiness report candidate axes do not match")
        passed = report.get("next_live_gate_passed") is True
        approved_authority = candidate_authority if passed else "none"
        approved_axes = candidate_axes if passed else NO_AXIS_AUTHORITY
    else:
        approved_authority = candidate_authority
        approved_axes = candidate_axes
    if (
        report.get("approved_authority") != approved_authority
        or report.get("axis_authority") != approved_axes
    ):
        errors.append("readiness report effective authority contradicts gate")
    return errors


def _authority_fields(
    candidate_authority: str,
    candidate_axes: AxisAuthority,
    passed: bool,
) -> AuthorityDecisionFields:
    approved_authority = candidate_authority if passed else "none"
    approved_axes = candidate_axes if passed else NO_AXIS_AUTHORITY
    return AuthorityDecisionFields(
        candidate_authority=candidate_authority,
        candidate_axis_authority=candidate_axes.copy(),
        approved_authority=approved_authority,
        axis_authority=approved_axes.copy(),
    )
