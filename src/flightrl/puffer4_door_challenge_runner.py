from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from flightrl.puffer4_door_challenge_evaluation import (
    load_matched_control_report,
    resolve_challenge_output,
)
from flightrl.puffer4_door_challenge_reporting import (
    build_door_challenge_report,
)
from flightrl.puffer4_door_challenge_specs import resolve_door_challenge
from flightrl.puffer4_door_eval_provenance import (
    FixedDoorEvaluationProvenanceCapture,
    write_fixed_door_evaluation_report,
)
from flightrl.puffer4_door_promotion_eval import (
    evaluate_promotion_door_policy,
)


def run_door_challenge_evaluation(
    *,
    bundle,
    policy,
    puffer_args: Mapping[str, Any],
    torch_pufferl,
    challenge: str,
    control_report: Path,
    output: Path | None,
    native_build_fingerprint: Mapping[str, Any],
    stream_contract: Mapping[str, Any],
    provenance_capture: FixedDoorEvaluationProvenanceCapture,
    puffer_root: Path,
    steps: int,
    seed: int,
    agents: int,
) -> tuple[dict[str, Any], Path]:
    trained_identity = bundle.trained_identity()
    control = load_matched_control_report(
        control_report,
        trained_identity=trained_identity,
        native_build_fingerprint=native_build_fingerprint,
        stream_contract=stream_contract,
        seed=seed,
        steps=steps,
        agents=agents,
    )
    resolved_output = resolve_challenge_output(
        bundle.checkpoint_path,
        challenge,
        control_report=control.path,
        lineage_report=bundle.report_path,
        requested=output,
    )
    challenge_args = deepcopy(dict(puffer_args))
    resolved_env, transform, raw_spec = resolve_door_challenge(
        challenge,
        challenge_args["env"],
        agent_count=agents,
    )
    challenge_args["env"] = resolved_env
    spec = dict(raw_spec)
    spec["resolved_environment_values"] = {
        key: resolved_env[key]
        for key in spec["environment_overrides"]
    }
    metrics = evaluate_promotion_door_policy(
        policy,
        challenge_args,
        torch_pufferl,
        steps=steps,
        seed=seed,
        camera_mask=False,
        agents=agents,
        observation_transform=transform,
    )
    report = build_door_challenge_report(
        output=resolved_output,
        trained_identity=trained_identity,
        lineage=bundle.lineage(),
        native_build_fingerprint=native_build_fingerprint,
        stream_contract=stream_contract,
        seed=seed,
        steps=steps,
        agents=agents,
        challenge_spec=spec,
        metrics=metrics,
        control=control,
    )
    report = write_fixed_door_evaluation_report(
        report=report,
        output=resolved_output,
        capture=provenance_capture,
        lineage_report=bundle.report_path,
        puffer_root=puffer_root,
        env_name=bundle.env_name,
        native_build_fingerprint=dict(native_build_fingerprint),
    )
    return report, resolved_output
