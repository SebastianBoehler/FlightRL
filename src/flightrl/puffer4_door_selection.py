from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from flightrl.puffer4_door_selection_inputs import (
    PromotionSelectionInput,
    ScreenSelectionInput,
    load_promotion_selection_input,
    load_screen_selection_input,
)
from flightrl.puffer4_door_selection_metrics import (
    ablation_delta,
    check,
    lower_is_better_regression_check,
    promotion_values,
    regression_check,
    screen_checks,
)


AUTHORITATIVE_V59_SUCCESS = 0.7911
SCREEN_SUCCESS_MINIMUM = AUTHORITATIVE_V59_SUCCESS + 0.05
REQUIRED_SCREEN_SEEDS = (11, 23, 47)


def build_fixed_door_selection_report(
    *,
    candidate_checkpoint: str | Path,
    candidate_report: str | Path,
    baseline_checkpoint: str | Path,
    baseline_report: str | Path,
    screens: Mapping[int, str | Path],
) -> dict[str, Any]:
    if set(screens) != set(REQUIRED_SCREEN_SEEDS):
        raise ValueError("selection requires screen reports for seeds 11, 23, and 47")
    candidate = load_promotion_selection_input(
        candidate_checkpoint,
        candidate_report,
    )
    baseline = load_promotion_selection_input(
        baseline_checkpoint,
        baseline_report,
    )
    screen_inputs = {
        seed: load_screen_selection_input(screens[seed], seed=seed)
        for seed in REQUIRED_SCREEN_SEEDS
    }
    _verify_matched_promotions(candidate, baseline)
    _verify_screen_lineage(candidate, baseline, screen_inputs)

    candidate_values = promotion_values(candidate)
    baseline_values = promotion_values(baseline)
    screening_checks = {
        str(seed): screen_checks(
            screen_inputs[seed],
            success_minimum=SCREEN_SUCCESS_MINIMUM,
        )
        for seed in REQUIRED_SCREEN_SEEDS
    }
    checks = {
        "mission_delta_at_least_0_05": check(
            candidate_values["mission_success"]
            - baseline_values["mission_success"]
            >= 0.05,
            candidate=candidate_values["mission_success"],
            baseline=baseline_values["mission_success"],
            delta=(
                candidate_values["mission_success"]
                - baseline_values["mission_success"]
            ),
            minimum_delta=0.05,
        ),
        "candidate_collision_at_most_0_03": check(
            candidate_values["collision"] <= 0.03,
            candidate=candidate_values["collision"],
            maximum=0.03,
        ),
        "outside_fov_regression_at_most_0_02": regression_check(
            candidate_values["outside_fov_success"],
            baseline_values["outside_fov_success"],
        ),
        "masked_camera_at_most_0_05": check(
            candidate_values["masked_success"] <= 0.05,
            candidate=candidate_values["masked_success"],
            maximum=0.05,
        ),
        "masked_camera_regression_at_most_0_02": (
            lower_is_better_regression_check(
                candidate_values["masked_success"],
                baseline_values["masked_success"],
            )
        ),
        "masked_collision_at_most_0_03": check(
            candidate_values["masked_collision"] <= 0.03,
            candidate=candidate_values["masked_collision"],
            maximum=0.03,
        ),
        "masked_collision_regression_at_most_0_02": (
            lower_is_better_regression_check(
                candidate_values["masked_collision"],
                baseline_values["masked_collision"],
            )
        ),
        "worst_marginal_regression_at_most_0_02": regression_check(
            candidate_values["worst_marginal_success"],
            baseline_values["worst_marginal_success"],
        ),
        "policy_latency_at_most_1_25x": check(
            candidate_values["policy_latency_p95_ms"]
            <= 1.25 * baseline_values["policy_latency_p95_ms"],
            candidate=candidate_values["policy_latency_p95_ms"],
            baseline=baseline_values["policy_latency_p95_ms"],
            maximum_ratio=1.25,
        ),
        "throughput_at_least_0_80x": check(
            candidate_values["throughput_sps"]
            >= 0.80 * baseline_values["throughput_sps"],
            candidate=candidate_values["throughput_sps"],
            baseline=baseline_values["throughput_sps"],
            minimum_ratio=0.80,
        ),
        "all_screen_seeds_pass": check(
            all(
                all(item["passed"] for item in seed_checks.values())
                for seed_checks in screening_checks.values()
            ),
            seeds=screening_checks,
        ),
    }
    selection_passed = all(item["passed"] for item in checks.values())
    cap_checks = {
        "success_at_least_0_70": candidate_values["cap_success"] >= 0.70,
        "outside_fov_at_least_0_65": (
            candidate_values["cap_outside_fov_success"] >= 0.65
        ),
        "collision_at_most_0_03": candidate_values["cap_collision"] <= 0.03,
    }
    report: dict[str, Any] = {
        "selection_schema": "flightrl.fixed_door.held_out_selection.v1",
        "selection_passed": selection_passed,
        "next_gate": "shadow_only",
        "real_shadow_evidence_present": False,
        "authoritative_v59_success": AUTHORITATIVE_V59_SUCCESS,
        "inputs": {
            "candidate": candidate.identity(),
            "baseline": baseline.identity(),
            "screens": {
                str(seed): screen_inputs[seed].identity()
                for seed in REQUIRED_SCREEN_SEEDS
            },
        },
        "matched_evaluation_contract": {
            "environment": dict(candidate.environment),
            "native_build_fingerprint": dict(candidate.native_fingerprint),
            "procedural_stream_contract": dict(candidate.stream_contract),
            "evidence_age_runtime_contract": dict(
                candidate.evidence_age_contract
            ),
        },
        "metrics": {
            "candidate": candidate_values,
            "baseline": baseline_values,
        },
        "selection_checks": checks,
        "screening_checks": screening_checks,
        "ablation_deltas": {
            "recurrence": ablation_delta(candidate, "recurrence"),
            "temporal": ablation_delta(candidate, "temporal"),
            "baseline": {
                "recurrence": ablation_delta(baseline, "recurrence"),
                "temporal": ablation_delta(baseline, "temporal"),
            },
        },
        "live_cap_simulation_ready": all(cap_checks.values()),
        "live_cap_simulation_checks": cap_checks,
    }
    if selection_passed:
        report["recommended_checkpoint"] = candidate.identity()["checkpoint"]
    return report


def write_exclusive_selection_report(
    output: str | Path,
    report: Mapping[str, Any],
    *,
    input_paths: tuple[str | Path, ...],
) -> Path:
    path = Path(output).resolve()
    aliases = {
        *(Path(item).resolve() for item in input_paths),
        *_bound_input_paths(report.get("inputs")),
    }
    if path in aliases:
        raise ValueError("selection output cannot alias an input")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite selection output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _bound_input_paths(value: object) -> set[Path]:
    if not isinstance(value, Mapping):
        return set()
    paths = {
        Path(path).resolve()
        for key, path in value.items()
        if key == "path" and isinstance(path, str)
    }
    for nested in value.values():
        paths.update(_bound_input_paths(nested))
    return paths


def _verify_matched_promotions(
    candidate: PromotionSelectionInput,
    baseline: PromotionSelectionInput,
) -> None:
    pairs = (
        ("evaluation environment", candidate.environment, baseline.environment),
        (
            "native build fingerprint",
            candidate.native_fingerprint,
            baseline.native_fingerprint,
        ),
        ("evaluation stream", candidate.stream_contract, baseline.stream_contract),
        (
            "evidence-age contract",
            candidate.evidence_age_contract,
            baseline.evidence_age_contract,
        ),
    )
    for label, left, right in pairs:
        if dict(left) != dict(right):
            raise ValueError(f"candidate and baseline {label} do not match")


def _verify_screen_lineage(
    candidate: PromotionSelectionInput,
    baseline: PromotionSelectionInput,
    screens: Mapping[int, ScreenSelectionInput],
) -> None:
    reference = screens[11]
    if (
        candidate.bundle.checkpoint_path != reference.bundle.checkpoint_path
        or candidate.bundle.lineage_report_path != reference.bundle.report_path
        or candidate.bundle.lineage_report_sha256
        != reference.bundle.report_sha256
    ):
        raise ValueError("candidate checkpoint does not match seed-11 screen lineage")
    reference_contracts = (
        reference.bundle.action_contract.to_report(),
        reference.bundle.policy_contract,
        reference.bundle.stream_contract,
        reference.parent,
        reference.budget,
    )
    for seed, screen in screens.items():
        values = (
            screen.bundle.action_contract.to_report(),
            screen.bundle.policy_contract,
            screen.bundle.stream_contract,
            screen.parent,
            screen.budget,
        )
        if values != reference_contracts:
            raise ValueError(f"screen seed {seed} contract, parent, or budget differs")
    if (
        candidate.bundle.action_contract != reference.bundle.action_contract
        or candidate.bundle.policy_contract != reference.bundle.policy_contract
        or candidate.bundle.stream_contract != reference.bundle.stream_contract
        or candidate.parent != reference.parent
        or baseline.parent != reference.parent
    ):
        raise ValueError("candidate, baseline, and screens do not share exact lineage")
