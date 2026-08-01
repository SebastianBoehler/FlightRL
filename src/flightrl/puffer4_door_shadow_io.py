from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from flightrl.hardware.config import CrazyflieHardwareConfig
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT as EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_door_live_evidence import (
    validate_fixed_door_live_evidence,
)
from flightrl.puffer4_door_shadow_csv import (
    SHADOW_CSV_REQUIRED_FIELDS as SHADOW_CSV_REQUIRED_FIELDS,
    read_shadow_csv_evidence as read_shadow_csv_evidence,
    shadow_identity_from_rows,
)
from flightrl.puffer4_door_shadow_identity import (
    SHADOW_IDENTITY_JSON_FIELD,
    SHADOW_IDENTITY_SHA256_FIELD,
    require_shadow_identity_matches_evidence,
)
from flightrl.puffer4_door_shadow_metrics import (
    MAX_FRAME_ASPECT_ERROR as MAX_FRAME_ASPECT_ERROR,
    MIN_SAMPLED_COVERAGE_S as MIN_SAMPLED_COVERAGE_S,
    MIN_SEARCH_PHASE_ROWS as MIN_SEARCH_PHASE_ROWS,
    MIN_SHADOW_FRAME_HEIGHT as MIN_SHADOW_FRAME_HEIGHT,
    MIN_SHADOW_FRAME_WIDTH as MIN_SHADOW_FRAME_WIDTH,
    MIN_TARGET_PHASE_ROWS as MIN_TARGET_PHASE_ROWS,
    detection_yaw_alignment,
    shadow_capture_contract,
    shadow_stream_metrics,
)
from flightrl.puffer4_door_shadow_projection import (
    summarize_fixed_door_shadow_projection,
)


REQUIRED_TELEMETRY = (
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.vz",
    "stateEstimate.roll",
    "stateEstimate.pitch",
    "stateEstimate.yaw",
    "gyro.x",
    "gyro.y",
    "gyro.z",
)
OPTIONAL_TELEMETRY = ("pm.vbat",)
TELEMETRY_VARIABLES = REQUIRED_TELEMETRY + OPTIONAL_TELEMETRY
SHADOW_LOG_PERIOD_MS = 100


def configure_shadow_logging(
    config: CrazyflieHardwareConfig,
) -> CrazyflieHardwareConfig:
    logging = replace(
        config.logging,
        period_ms=SHADOW_LOG_PERIOD_MS,
        variables=TELEMETRY_VARIABLES,
    )
    return replace(config, logging=logging)


def require_telemetry_contract(variables: Sequence[str]) -> None:
    missing = sorted(set(REQUIRED_TELEMETRY) - set(variables))
    if missing:
        raise RuntimeError(f"Crazyflie telemetry contract missing: {missing}")


def telemetry_csv_fields(values: Mapping[str, float]) -> dict[str, float | None]:
    return {
        f"telemetry_{name.replace('.', '_')}": values.get(name)
        for name in TELEMETRY_VARIABLES
    }


def summarize_shadow_rows(
    rows: list[dict],
    *,
    checkpoint: str | Path,
    training_report: str | Path,
    simulation_gate: dict,
    dropped_frames: int,
) -> dict:
    evidence = validate_fixed_door_live_evidence(checkpoint, training_report)
    bundle = evidence.bundle
    identity = shadow_identity_from_rows(rows)
    require_shadow_identity_matches_evidence(identity, evidence)
    if any(
        row.get("controls_drone") is not False
        or row.get("monitor_only") is not True
        for row in rows
    ):
        raise ValueError("shadow summary rows must remain non-actuating")
    forward = np.asarray([row["action_forward"] for row in rows])
    yaw = np.asarray([row["action_yaw"] for row in rows])
    phases = {
        phase: sum(row["phase"] == phase for row in rows)
        for phase in ("search", "track", "approach", "recover")
    }
    yaw_samples, yaw_alignment = detection_yaw_alignment(rows)
    projection_samples, projection_alignment = detection_yaw_alignment(
        rows,
        yaw_field="yaw_only_projected_yawrate_deg_s",
    )
    grounding_ages = [
        float(row["grounding_age_s"])
        for row in rows
        if row.get("grounding_age_s") is not None
    ]
    capture = shadow_capture_contract(rows, phases)
    stream_metrics = shadow_stream_metrics(rows)
    if stream_metrics["stream_dropped_frames"] != dropped_frames:
        raise ValueError(
            "shadow stream drop counter does not match captured rows"
        )
    action_contract = bundle.action_contract.to_report()
    policy_contract = bundle.policy_contract
    return {
        "checkpoint": str(bundle.checkpoint_path),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "training_report": str(bundle.report_path),
        "training_report_sha256": bundle.report_sha256,
        "evaluation_report": str(bundle.report_path),
        "evaluation_report_sha256": bundle.report_sha256,
        "lineage_report": str(bundle.lineage_report_path),
        "lineage_report_sha256": bundle.lineage_report_sha256,
        "action_contract_id": action_contract.get("contract_id"),
        "action_contract_sha256": action_contract.get("sha256"),
        "policy_contract_id": policy_contract.get("contract_id"),
        "policy_contract_sha256": policy_contract.get("sha256"),
        "evidence_age_runtime_contract": EVIDENCE_AGE_CONTRACT.to_report(),
        "shadow_run_identity": dict(identity.payload),
        SHADOW_IDENTITY_JSON_FIELD: identity.canonical_json,
        SHADOW_IDENTITY_SHA256_FIELD: identity.sha256,
        "controls_drone": False,
        "monitor_only": True,
        "simulation_gate_passed": bool(simulation_gate.get("passed")),
        "rows": len(rows),
        "dropped_frames": dropped_frames,
        "phase_counts": phases,
        "detection_rows": sum(bool(row["target_detected"]) for row in rows),
        "detection_yaw_alignment_samples": yaw_samples,
        "detection_yaw_sign_accuracy": yaw_alignment,
        "yaw_only_detection_yaw_alignment_samples": projection_samples,
        "yaw_only_detection_yaw_sign_accuracy": projection_alignment,
        "all_actions_finite": bool(
            np.isfinite(forward).all() and np.isfinite(yaw).all()
        ),
        "forward_mean": float(forward.mean()),
        "forward_p95": float(np.percentile(forward, 95)),
        "abs_yaw_p95": float(np.percentile(np.abs(yaw), 95)),
        "inference_ms_p95": float(
            np.percentile([row["inference_ms"] for row in rows], 95)
        ),
        "grounding_age_s_p95": (
            None
            if not grounding_ages
            else float(np.percentile(grounding_ages, 95))
        ),
        **capture,
        **stream_metrics,
        **summarize_fixed_door_shadow_projection(
            rows,
            bundle.action_contract,
        ),
        "next_gate": "review real shadow trace; no learned control authority",
    }
