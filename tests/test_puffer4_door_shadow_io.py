import json
from pathlib import Path

import pytest

from flightrl.hardware.config import CrazyflieHardwareConfig
from flightrl.puffer4_door_live_evidence import (
    validate_fixed_door_live_evidence,
)
from flightrl.puffer4_door_shadow_identity import (
    APPROVED_SHADOW_DETECTOR_MODEL_ID,
    APPROVED_SHADOW_DEVICE,
    APPROVED_SHADOW_HARDWARE_CONFIG,
    APPROVED_SHADOW_PROMPT,
    APPROVED_SHADOW_THRESHOLD,
    build_fixed_door_shadow_identity,
)
from flightrl.puffer4_door_shadow_io import (
    REQUIRED_TELEMETRY,
    SHADOW_LOG_PERIOD_MS,
    TELEMETRY_VARIABLES,
    configure_shadow_logging,
    detection_yaw_alignment,
    require_telemetry_contract,
    summarize_shadow_rows,
    telemetry_csv_fields,
)
from flightrl.puffer4_door_shadow_projection import (
    bind_fixed_door_shadow_rows,
)


ROOT = Path(__file__).resolve().parents[1]
V59_CHECKPOINT = (
    ROOT
    / "artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m"
    / "flightrl_fixed_door_d1_seed11_1048576.bin"
)
V59_REPORT = V59_CHECKPOINT.with_suffix(".reevaluation.json")


def test_shadow_logging_uses_only_required_low_rate_contract() -> None:
    source = CrazyflieHardwareConfig()

    configured = configure_shadow_logging(source)

    assert configured.logging.variables == TELEMETRY_VARIABLES
    assert configured.logging.period_ms == SHADOW_LOG_PERIOD_MS
    assert source.logging.variables != TELEMETRY_VARIABLES


def test_shadow_contract_allows_missing_optional_battery() -> None:
    require_telemetry_contract(REQUIRED_TELEMETRY)


def test_shadow_csv_fields_preserve_exact_values() -> None:
    fields = telemetry_csv_fields(
        {
            "stateEstimate.x": 1.25,
            "pm.vbat": 3.8,
        }
    )

    assert fields["telemetry_stateEstimate_x"] == 1.25
    assert fields["telemetry_pm_vbat"] == 3.8
    assert fields["telemetry_gyro_z"] is None


def test_shadow_yaw_alignment_requires_nonzero_correct_sign() -> None:
    left_detection = json.dumps(
        {
            "box": {
                "x_min": 0.0,
                "x_max": 0.2,
            }
        }
    )

    samples, accuracy = detection_yaw_alignment(
        [
            {"detection": left_detection, "action_yaw": 0.0},
            {"detection": left_detection, "action_yaw": 0.1},
        ]
    )

    assert samples == 2
    assert accuracy == 0.5


def test_shadow_summary_binds_report_and_sampled_capture_contract(
) -> None:
    evidence = validate_fixed_door_live_evidence(V59_CHECKPOINT, V59_REPORT)
    identity = build_fixed_door_shadow_identity(
        evidence,
        prompt=APPROVED_SHADOW_PROMPT,
        detector_model_id=APPROVED_SHADOW_DETECTOR_MODEL_ID,
        threshold=APPROVED_SHADOW_THRESHOLD,
        device=APPROVED_SHADOW_DEVICE,
        hardware_config=APPROVED_SHADOW_HARDWARE_CONFIG,
    )
    rows = bind_fixed_door_shadow_rows(
        _healthy_shadow_rows(),
        identity,
        evidence.bundle.action_contract,
    )

    summary = summarize_shadow_rows(
        rows,
        checkpoint=V59_CHECKPOINT,
        training_report=V59_REPORT,
        simulation_gate={"passed": True},
        dropped_frames=0,
    )

    assert summary["training_report_sha256"] == (
        "b919e4f9951ad28904ce6cc7ee9b7a0f7b76ee70fba387e673bcda27a9bdcbbc"
    )
    assert summary["checkpoint_sha256"] == (
        "f676d12b9d37c27f4cc62f99beceec8f30e74c88be8564cb242c23755e202cce"
    )
    assert summary["evaluation_report"] == str(V59_REPORT.resolve())
    assert (
        summary["action_contract_id"]
        == "fixed-door-v59-legacy-physics-yaw-v1"
    )
    assert (
        summary["action_contract_sha256"]
        == "e666cf9c708b43e344355bad1c8b8f4c62826a79a26046839677e454879a80b5"
    )
    assert summary["policy_contract_id"] == "fixed-door-recurrent-policy-v1"
    assert summary["policy_contract_sha256"] == (
        "ad6fa58f50a1c0754d572643a9d7affe65f3e73d4d814c51030c733588ef8058"
    )
    assert summary["sampled_coverage_s"] == pytest.approx(20.0)
    assert summary["frame_width"] == 324
    assert summary["frame_height"] == 244
    assert summary["frame_geometry_contract_passed"] is True
    assert summary["phase_coverage_passed"] is True
    assert summary["shadow_run_identity"] == identity.payload
    assert summary["shadow_run_identity_sha256"] == identity.sha256
    assert summary["yaw_only_projection_contract_passed"] is True
    assert summary["yaw_only_projected_forward_abs_max_m_s"] == 0.0
    assert summary["yaw_only_projected_abs_yawrate_max_deg_s"] == 8.0
    assert summary["executed_previous_action_abs_max"] == 0.0
    assert summary["stream_dropped_frames"] == 0
    assert summary["frame_index_gap_count"] == 0
    assert summary["grounding_inference_ms_p95"] == 500.0
    assert summary["grounding_result_rate_hz"] == pytest.approx(10.0)


def test_shadow_summary_rejects_rows_relabelled_to_another_checkpoint() -> None:
    evidence = validate_fixed_door_live_evidence(V59_CHECKPOINT, V59_REPORT)
    identity = build_fixed_door_shadow_identity(
        evidence,
        prompt=APPROVED_SHADOW_PROMPT,
        detector_model_id=APPROVED_SHADOW_DETECTOR_MODEL_ID,
        threshold=APPROVED_SHADOW_THRESHOLD,
        device=APPROVED_SHADOW_DEVICE,
        hardware_config=APPROVED_SHADOW_HARDWARE_CONFIG,
    )
    rows = bind_fixed_door_shadow_rows(
        _healthy_shadow_rows(),
        identity,
        evidence.bundle.action_contract,
    )
    payload = dict(identity.payload)
    payload["checkpoint"] = {
        **identity.payload["checkpoint"],
        "path": str((ROOT / "other.bin").resolve()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    import hashlib

    for row in rows:
        row["shadow_run_identity_json"] = encoded
        row["shadow_run_identity_sha256"] = hashlib.sha256(
            encoded.encode()
        ).hexdigest()

    with pytest.raises(ValueError, match="checkpoint"):
        summarize_shadow_rows(
            rows,
            checkpoint=V59_CHECKPOINT,
            training_report=V59_REPORT,
            simulation_gate={"passed": True},
            dropped_frames=0,
        )


def test_shadow_summary_rejects_actuating_row_under_monitor_identity() -> None:
    evidence = validate_fixed_door_live_evidence(V59_CHECKPOINT, V59_REPORT)
    identity = build_fixed_door_shadow_identity(
        evidence,
        prompt=APPROVED_SHADOW_PROMPT,
        detector_model_id=APPROVED_SHADOW_DETECTOR_MODEL_ID,
        threshold=APPROVED_SHADOW_THRESHOLD,
        device=APPROVED_SHADOW_DEVICE,
        hardware_config=APPROVED_SHADOW_HARDWARE_CONFIG,
    )
    rows = bind_fixed_door_shadow_rows(
        _healthy_shadow_rows(),
        identity,
        evidence.bundle.action_contract,
    )
    rows[0]["controls_drone"] = True

    with pytest.raises(ValueError, match="non-actuating"):
        summarize_shadow_rows(
            rows,
            checkpoint=V59_CHECKPOINT,
            training_report=V59_REPORT,
            simulation_gate={"passed": True},
            dropped_frames=0,
        )


def _healthy_shadow_rows() -> list[dict]:
    left_detection = json.dumps(
        {
            "box": {
                "x_min": 0.0,
                "x_max": 0.2,
            }
        }
    )
    rows = []
    for index in range(200):
        detected = index >= 20
        rows.append(
            {
                "frame_index": index,
                "frame_host_time_s": index / 10.0,
                "frame_width": 324,
                "frame_height": 244,
                "action_forward": 0.1,
                "action_yaw": 0.1,
                "controls_drone": False,
                "monitor_only": True,
                "phase": "track" if detected else "search",
                "target_detected": detected,
                "detection": left_detection if detected else None,
                "inference_ms": 1.0,
                "grounding_age_s": 0.5 if detected else None,
                "grounding_inference_ms": 500.0,
                "grounding_result_frame_index": index,
                "stream_dropped_frames": 0,
            }
        )
    return rows
