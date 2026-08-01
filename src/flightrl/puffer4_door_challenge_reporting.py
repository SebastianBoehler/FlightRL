from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from flightrl.puffer4_door_challenge_evaluation import MatchedControlReport
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)


def build_door_challenge_report(
    *,
    output: Path,
    trained_identity: Mapping[str, Any],
    lineage: Mapping[str, Any],
    native_build_fingerprint: Mapping[str, Any],
    stream_contract: Mapping[str, Any],
    seed: int,
    steps: int,
    agents: int,
    challenge_spec: Mapping[str, Any],
    metrics: Mapping[str, Any],
    control: MatchedControlReport,
) -> dict[str, Any]:
    if challenge_spec.get("combined_with_other_challenges") is not False:
        raise ValueError("challenge spec does not prove a single intervention")
    name = challenge_spec.get("name")
    if not isinstance(name, str):
        raise ValueError("challenge spec has no name")
    control_summary = _comparison_metrics(control.metrics)
    challenge_summary = _comparison_metrics(metrics)
    common = control_summary.keys() & challenge_summary.keys()
    return {
        "evaluation_schema": "flightrl.fixed_door.challenge.v1",
        "trained_identity": dict(trained_identity),
        "evaluation_identity": {
            "kind": "fixed_door_challenge",
            "schema_version": 1,
            "report": {"path": str(output.resolve())},
            "environment": {
                "name": _environment_name(trained_identity),
                "agents": agents,
                "steps": steps,
                "seed": seed,
            },
            "action_contract_sha256": _contract_sha(
                trained_identity, "action_contract"
            ),
            "policy_contract_sha256": _contract_sha(
                trained_identity, "policy_contract"
            ),
            "procedural_stream_contract": dict(stream_contract),
            "evidence_age_runtime_contract": (
                FIXED_DOOR_EVIDENCE_AGE_CONTRACT.to_report()
            ),
            "native_build_fingerprint": dict(native_build_fingerprint),
        },
        "lineage": dict(lineage),
        "matched_control": {
            "report": {
                "path": str(control.path),
                "sha256": control.sha256,
            },
            "environment": dict(control.environment),
            "metrics": dict(control.metrics),
        },
        "challenge": {
            "resolved_single_variable": dict(challenge_spec),
            "metrics": dict(metrics),
        },
        "comparison": {
            "control": control_summary,
            "challenge": challenge_summary,
            "delta_challenge_minus_control": {
                key: challenge_summary[key] - control_summary[key]
                for key in sorted(common)
            },
        },
        "limitations": [
            str(challenge_spec.get("limitation")),
            (
                "The intervention changes closed-loop actions, so matched "
                "seeds do not imply identical post-intervention trajectories."
            ),
            "Diagnostic challenge evidence cannot authorize physical control.",
        ],
        "simulation_gate": False,
        "live_eligible": False,
    }


def _comparison_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {
        key: float(metrics[key])
        for key in (
            "success_rate",
            "outside_fov_success_rate",
            "collision_rate",
            "forward_action_mean",
            "yaw_proposal_abs_p95",
            "yaw_action_p95",
        )
        if isinstance(metrics.get(key), (int, float))
    }
    performance = metrics.get("performance")
    if isinstance(performance, Mapping):
        nested = (
            ("closed_loop_agent_steps_per_second", None),
            ("closed_loop_p95_ms", "closed_loop_batch_ms"),
            ("policy_forward_p95_ms", "policy_forward_batch_ms"),
        )
        for output_key, source_key in nested:
            value = (
                performance.get(output_key)
                if source_key is None
                else _nested_p95(performance.get(source_key))
            )
            if isinstance(value, (int, float)):
                result[output_key] = float(value)
    return result


def _nested_p95(value: object) -> object:
    return value.get("p95") if isinstance(value, Mapping) else None


def _contract_sha(identity: Mapping[str, Any], key: str) -> str:
    contract = identity.get(key)
    digest = contract.get("sha256") if isinstance(contract, Mapping) else None
    if not isinstance(digest, str):
        raise ValueError(f"trained {key} has no SHA-256")
    return digest


def _environment_name(identity: Mapping[str, Any]) -> str:
    environment = identity.get("environment")
    name = environment.get("name") if isinstance(environment, Mapping) else None
    if not isinstance(name, str):
        raise ValueError("trained environment has no name")
    return name
