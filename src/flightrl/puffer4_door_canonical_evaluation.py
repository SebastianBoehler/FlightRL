from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from flightrl.puffer4_door_contract import (
    FIXED_DOOR_LIVE_SAFETY_CONTRACT,
)
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_door_eval_provenance import (
    FixedDoorEvaluationProvenanceCapture,
    write_fixed_door_evaluation_report,
)
from flightrl.puffer4_door_promotion_eval import (
    build_recurrence_reset_ablation,
    evaluate_promotion_door_policy,
)
from flightrl.puffer4_door_temporal_ablation import (
    TEMPORAL_ORDER_ABLATION_SEED,
    DoorTemporalOrderScrambler,
    build_temporal_order_ablation,
)
from flightrl.puffer4_door_training import fixed_door_gate


def run_canonical_door_evaluation(
    *,
    bundle,
    policy,
    puffer_args: Mapping[str, Any],
    torch_pufferl,
    output: Path,
    native_build_fingerprint: Mapping[str, Any],
    stream_contract: Mapping[str, Any],
    provenance_capture: FixedDoorEvaluationProvenanceCapture,
    puffer_root: Path,
    steps: int,
    seed: int,
    agents: int,
    live_yaw_cap_challenge: bool,
) -> tuple[dict[str, Any], Path]:
    common = {
        "steps": steps,
        "seed": seed,
        "agents": agents,
    }
    full = evaluate_promotion_door_policy(
        policy,
        puffer_args,
        torch_pufferl,
        camera_mask=False,
        **common,
    )
    masked = evaluate_promotion_door_policy(
        policy,
        puffer_args,
        torch_pufferl,
        camera_mask=True,
        **common,
    )
    reset_each_step = evaluate_promotion_door_policy(
        policy,
        puffer_args,
        torch_pufferl,
        camera_mask=False,
        recurrent_mode="reset_each_step",
        **common,
    )
    recurrence_ablation = build_recurrence_reset_ablation(
        full,
        reset_each_step,
    )
    temporal_order = evaluate_promotion_door_policy(
        policy,
        puffer_args,
        torch_pufferl,
        camera_mask=False,
        temporal_order_seed=TEMPORAL_ORDER_ABLATION_SEED,
        **common,
    )
    temporal_ablation = build_temporal_order_ablation(
        full,
        temporal_order,
        DoorTemporalOrderScrambler(
            agent_count=agents,
            seed=TEMPORAL_ORDER_ABLATION_SEED,
        ),
    )
    report = _build_canonical_report(
        bundle=bundle,
        output=output,
        native_build_fingerprint=native_build_fingerprint,
        stream_contract=stream_contract,
        steps=steps,
        seed=seed,
        agents=agents,
        full=full,
        masked=masked,
        reset_each_step=reset_each_step,
        recurrence_ablation=recurrence_ablation,
        temporal_order=temporal_order,
        temporal_ablation=temporal_ablation,
    )
    if live_yaw_cap_challenge:
        report["live_yaw_cap_challenge"] = _run_live_yaw_cap_challenge(
            bundle=bundle,
            policy=policy,
            puffer_args=puffer_args,
            torch_pufferl=torch_pufferl,
            common=common,
        )
    written = write_fixed_door_evaluation_report(
        report=report,
        output=output,
        capture=provenance_capture,
        lineage_report=bundle.report_path,
        puffer_root=puffer_root,
        env_name=bundle.env_name,
        native_build_fingerprint=dict(native_build_fingerprint),
    )
    return written, output.resolve()


def _build_canonical_report(
    *,
    bundle,
    output: Path,
    native_build_fingerprint: Mapping[str, Any],
    stream_contract: Mapping[str, Any],
    steps: int,
    seed: int,
    agents: int,
    full: dict,
    masked: dict,
    reset_each_step: dict,
    recurrence_ablation: dict,
    temporal_order: dict,
    temporal_ablation: dict,
) -> dict[str, Any]:
    finite_runs = (full, masked, reset_each_step)
    return {
        "evaluation_schema": "flightrl.fixed_door.promotion.v3",
        "trained_identity": bundle.trained_identity(),
        "evaluation_identity": {
            "kind": "fixed_door_promotion",
            "schema_version": 1,
            "report": {"path": str(output.resolve())},
            "environment": {
                "name": bundle.env_name,
                "agents": agents,
                "steps_per_condition": steps,
                "seed": seed,
            },
            "action_contract_sha256": bundle.action_contract.sha256(),
            "policy_contract_sha256": bundle.policy_contract["sha256"],
            "procedural_stream_contract": dict(stream_contract),
            "evidence_age_runtime_contract": (
                FIXED_DOOR_EVIDENCE_AGE_CONTRACT.to_report()
            ),
            "native_build_fingerprint": dict(native_build_fingerprint),
        },
        "lineage": bundle.lineage(),
        "full_camera": full,
        "masked_camera": masked,
        "recurrence_reset_ablation": recurrence_ablation,
        "temporal_order_ablation": temporal_ablation,
        "simulation_gate": fixed_door_gate(full, masked),
        "promotion_evidence": {
            "all_default_runs_complete": all(
                run["status"] == "complete" for run in finite_runs
            ),
            "all_default_outputs_finite": all(
                run["finite_outputs"]["passed"] for run in finite_runs
            ),
            "temporal_order_run_complete": temporal_order["status"] == "complete",
            "temporal_order_outputs_finite": temporal_order[
                "finite_outputs"
            ]["passed"],
            "full_camera_worst_marginal_group": full[
                "marginal_groups"
            ].get("worst_marginal_group"),
        },
    }


def _run_live_yaw_cap_challenge(
    *,
    bundle,
    policy,
    puffer_args: Mapping[str, Any],
    torch_pufferl,
    common: Mapping[str, int],
) -> dict[str, Any]:
    live_limit = FIXED_DOOR_LIVE_SAFETY_CONTRACT.max_yawrate_deg_s
    policy_limit = bundle.action_contract.max_yawrate_deg_s
    normalized_limit = FIXED_DOOR_LIVE_SAFETY_CONTRACT.normalized_yaw_limit(
        bundle.action_contract
    )
    metrics = evaluate_promotion_door_policy(
        policy,
        puffer_args,
        torch_pufferl,
        camera_mask=False,
        yaw_abs_limit_normalized=normalized_limit,
        **common,
    )
    return {
        "label": "live_yaw_cap_only",
        "policy_max_yawrate_deg_s": policy_limit,
        "live_max_yawrate_deg_s": live_limit,
        "normalized_yaw_limit": normalized_limit,
        "condition": {
            "camera": "full",
            "recurrent_mode": "carried",
            "forward_action": "unchanged",
            "combined_with_other_ablation": False,
        },
        "metrics": metrics,
    }
