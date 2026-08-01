from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

from flightrl.puffer4_door_selection_inputs import (
    PromotionSelectionInput,
    ScreenSelectionInput,
)


def promotion_values(evidence: PromotionSelectionInput) -> dict[str, float]:
    performance = _mapping(evidence.full.get("performance"), "performance")
    latency = _mapping(
        performance.get("policy_forward_batch_ms"),
        "policy latency",
    )
    cap = evidence.live_cap
    return {
        "mission_success": number(evidence.full["success_rate"]),
        "outside_fov_success": number(
            evidence.full["outside_fov_success_rate"]
        ),
        "collision": number(evidence.full["collision_rate"]),
        "masked_success": number(evidence.masked["success_rate"]),
        "masked_collision": number(evidence.masked["collision_rate"]),
        "worst_marginal_success": _worst_marginal(evidence.full),
        "policy_latency_p95_ms": _positive(latency.get("p95"), "policy p95"),
        "throughput_sps": _positive(
            performance.get("closed_loop_agent_steps_per_second"),
            "throughput",
        ),
        "cap_success": number(cap["success_rate"]),
        "cap_outside_fov_success": number(cap["outside_fov_success_rate"]),
        "cap_collision": number(cap["collision_rate"]),
    }


def screen_checks(
    evidence: ScreenSelectionInput,
    *,
    success_minimum: float,
) -> dict[str, dict[str, Any]]:
    return {
        "mission_at_least_v59_plus_0_05": check(
            number(evidence.full["success_rate"]) >= success_minimum,
            value=number(evidence.full["success_rate"]),
            minimum=success_minimum,
        ),
        "collision_at_most_0_03": check(
            number(evidence.full["collision_rate"]) <= 0.03,
            value=number(evidence.full["collision_rate"]),
            maximum=0.03,
        ),
        "masked_at_most_0_05": check(
            number(evidence.masked["success_rate"]) <= 0.05,
            value=number(evidence.masked["success_rate"]),
            maximum=0.05,
        ),
    }


def ablation_delta(
    evidence: PromotionSelectionInput,
    kind: str,
) -> dict[str, float]:
    run = evidence.recurrence if kind == "recurrence" else evidence.temporal
    return {
        key: number(run[key]) - number(evidence.full[key])
        for key in (
            "success_rate",
            "outside_fov_success_rate",
            "collision_rate",
        )
    }


def regression_check(candidate: float, baseline: float) -> dict[str, Any]:
    return check(
        candidate >= baseline - 0.02,
        candidate=candidate,
        baseline=baseline,
        delta=candidate - baseline,
        minimum_delta=-0.02,
    )


def lower_is_better_regression_check(
    candidate: float,
    baseline: float,
) -> dict[str, Any]:
    return check(
        candidate <= baseline + 0.02,
        candidate=candidate,
        baseline=baseline,
        delta=candidate - baseline,
        maximum_delta=0.02,
    )


def check(passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **evidence}


def number(value: object) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError("selection metric must be finite")
    return parsed


def _worst_marginal(run: Mapping[str, Any]) -> float:
    groups = _mapping(run.get("marginal_groups"), "marginal groups")
    worst = _mapping(groups.get("worst_marginal_group"), "worst marginal group")
    if number(worst.get("support")) <= 0.0:
        raise ValueError("worst marginal group has no support")
    return number(worst.get("conditional_success_rate"))


def _positive(value: object, label: str) -> float:
    parsed = number(value)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing or invalid")
    return value
