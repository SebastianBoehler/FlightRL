from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from flightrl.puffer4_door_contract import (
    FIXED_DOOR_LIVE_SAFETY_CONTRACT,
)
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_door_readiness import (
    bind_fixed_door_readiness_identity,
    build_fixed_door_yaw_readiness,
    load_fixed_door_yaw_readiness,
)
from flightrl.puffer4_door_live_evidence import (
    validate_fixed_door_live_evidence,
)
from flightrl.puffer4_door_shadow_detector_contract import (
    APPROVED_SHADOW_DETECTOR_MODEL_ID,
    APPROVED_SHADOW_DEVICE,
    APPROVED_SHADOW_HARDWARE_CONFIG,
    APPROVED_SHADOW_PROMPT,
    APPROVED_SHADOW_THRESHOLD,
)
from flightrl.puffer4_door_shadow_identity import (
    build_fixed_door_shadow_identity,
)
from flightrl.puffer4_door_shadow_io import summarize_shadow_rows
from flightrl.puffer4_door_shadow_projection import (
    bind_fixed_door_shadow_rows,
)
from flightrl.semantic.readiness import write_readiness
from flightrl.semantic.readiness import file_sha256
from fixed_door_promotion_fixture import (
    write_test_lineage,
    write_test_promotion,
)


ROOT = Path(__file__).resolve().parents[1]
V59_CHECKPOINT = (
    ROOT
    / "artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m"
    / "flightrl_fixed_door_d1_seed11_1048576.bin"
)
V59_REPORT = V59_CHECKPOINT.with_suffix(".reevaluation.json")


def test_fixed_door_readiness_requires_matching_real_shadow(tmp_path) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)

    passed = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )
    readiness_path = tmp_path / "readiness.json"
    readiness = write_readiness(
        readiness_path,
        bind_fixed_door_readiness_identity(passed, readiness_path),
    )
    assert load_fixed_door_yaw_readiness(
        readiness,
        checkpoint,
        simulation,
    )["next_live_gate_passed"] is True

    mismatched_shadow = json.loads(shadow.read_text())
    mismatched_shadow["checkpoint"] = str((tmp_path / "other.bin").resolve())
    shadow.write_text(json.dumps(mismatched_shadow))
    rejected = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )

    assert passed["next_live_gate_passed"] is True
    assert passed["translation_authority_passed"] is False
    assert passed["limits"]["detected_abs_yawrate_deg_s"] == 8.0
    assert passed["limits"]["min_height_m"] == 0.20
    assert passed["limits"]["max_height_m"] == 0.80
    assert passed["limits"]["max_duration_s"] == 15.0
    assert (
        passed["live_safety_contract"]
        == FIXED_DOOR_LIVE_SAFETY_CONTRACT.to_report()
    )
    assert rejected["next_live_gate_passed"] is False


def test_fixed_door_yaw_readiness_rejects_flat_training_report(
    tmp_path,
) -> None:
    checkpoint, flat_report = write_test_lineage(tmp_path)

    with pytest.raises(ValueError, match="promotion.v3"):
        build_fixed_door_yaw_readiness(
            checkpoint,
            flat_report,
            tmp_path / "missing-summary.json",
            tmp_path / "missing-shadow.csv",
        )


def test_fixed_door_yaw_readiness_rejects_grandfathered_v59(
    tmp_path,
) -> None:
    evidence = validate_fixed_door_live_evidence(
        V59_CHECKPOINT,
        V59_REPORT,
    )
    identity = _shadow_identity(evidence)
    rows = bind_fixed_door_shadow_rows(
        _healthy_shadow_rows(),
        identity,
        evidence.bundle.action_contract,
    )
    summary = tmp_path / "v59.summary.json"
    summary.write_text(
        json.dumps(
            summarize_shadow_rows(
                rows,
                checkpoint=V59_CHECKPOINT,
                training_report=V59_REPORT,
                simulation_gate={"passed": True},
                dropped_frames=0,
            )
        )
    )
    shadow_csv = tmp_path / "v59.csv"
    _write_csv(shadow_csv, rows)

    with pytest.raises(ValueError, match="promotion.v3"):
        build_fixed_door_yaw_readiness(
            V59_CHECKPOINT,
            V59_REPORT,
            summary,
            shadow_csv,
        )


def test_fixed_door_readiness_binds_approved_evidence_age_runtime(
    tmp_path,
) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)

    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )
    shadow_report = json.loads(shadow.read_text())

    assert (
        shadow_report["evidence_age_runtime_contract"]
        == FIXED_DOOR_EVIDENCE_AGE_CONTRACT.to_report()
    )
    assert (
        report["evidence_age_runtime_contract"]
        == FIXED_DOOR_EVIDENCE_AGE_CONTRACT.to_report()
    )
    assert report["shadow_evidence_age_binding_passed"] is True


@pytest.mark.parametrize(
    ("metric", "value"),
    (
        ("success_rate", 0.69),
        ("outside_fov_success_rate", 0.64),
        ("collision_rate", 0.031),
    ),
)
def test_fixed_door_readiness_requires_live_yaw_cap_metrics(
    tmp_path,
    metric: str,
    value: float,
) -> None:
    checkpoint, simulation = write_test_promotion(tmp_path)
    payload = json.loads(simulation.read_text())
    payload["live_yaw_cap_challenge"]["metrics"][metric] = value
    simulation.write_text(json.dumps(payload))
    checkpoint, simulation, shadow, shadow_csv = _shadow_artifacts(
        checkpoint,
        simulation,
        tmp_path,
    )

    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )

    assert report["sim_live_yaw_cap_gate_passed"] is False
    assert report["replay_yaw_gate_passed"] is True
    assert report["next_live_gate_passed"] is False


@pytest.mark.parametrize(
    "fault",
    ("detector-latency", "detector-cadence", "stream-drops"),
)
def test_fixed_door_readiness_recomputes_shadow_stream_health(
    tmp_path,
    fault: str,
) -> None:
    checkpoint, simulation = write_test_promotion(tmp_path)
    rows = _healthy_shadow_rows()
    if fault == "detector-latency":
        rows = [
            row | {"grounding_inference_ms": 751.0}
            for row in rows
        ]
    elif fault == "detector-cadence":
        rows = [
            row | {"grounding_result_frame_index": index // 20}
            for index, row in enumerate(rows)
        ]
    else:
        rows = [
            row | {"stream_dropped_frames": 6}
            for row in rows
        ]
    checkpoint, simulation, shadow, shadow_csv = _shadow_artifacts(
        checkpoint,
        simulation,
        tmp_path,
        rows=rows,
    )

    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )

    assert report["shadow_csv_binding_passed"] is True
    assert report["replay_yaw_gate_passed"] is False
    assert report["next_live_gate_passed"] is False


def test_fixed_door_readiness_rejects_rehashed_evidence_age_drift(
    tmp_path,
) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)
    summary = json.loads(shadow.read_text())
    encoded = summary["evidence_age_runtime_contract"]
    encoded["maximum_evidence_age_s"] = 1.1
    payload = {key: value for key, value in encoded.items() if key != "sha256"}
    encoded["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    shadow.write_text(json.dumps(summary))

    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )

    assert report["shadow_evidence_age_binding_passed"] is False
    assert report["next_live_gate_passed"] is False


def test_fixed_door_readiness_binds_exact_simulation_report(tmp_path) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)
    other_simulation = tmp_path / "other-simulation.json"
    other_simulation.write_bytes(simulation.read_bytes())

    with pytest.raises(ValueError, match="evaluation report path"):
        build_fixed_door_yaw_readiness(
            checkpoint,
            other_simulation,
            shadow,
            shadow_csv,
        )


def test_fixed_door_readiness_rejects_self_consistent_csv_relabel(
    tmp_path,
) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)
    with shadow_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads(rows[0]["shadow_run_identity_json"])
    payload["checkpoint"]["path"] = str((tmp_path / "other.bin").resolve())
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    for row in rows:
        row["shadow_run_identity_json"] = encoded
        row["shadow_run_identity_sha256"] = digest
    _write_csv(shadow_csv, rows)

    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )

    assert report["shadow_run_identity_binding_passed"] is False
    assert report["next_live_gate_passed"] is False


@pytest.mark.parametrize("fault", ("short", "geometry", "phase"))
def test_fixed_door_readiness_recomputes_capture_contract_from_csv(
    tmp_path,
    fault: str,
) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)
    rows = _bound_rows(
        checkpoint,
        simulation,
        _healthy_shadow_rows(),
    )
    if fault == "short":
        rows = rows[:100]
    elif fault == "geometry":
        rows = [row | {"frame_width": 64} for row in rows]
    else:
        rows = [row | {"phase": "track"} for row in rows]
    _write_csv(shadow_csv, rows)

    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )

    assert report["next_live_gate_passed"] is False


def test_fixed_door_readiness_rejects_missing_csv_contract_column(
    tmp_path,
) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)
    rows = _bound_rows(
        checkpoint,
        simulation,
        _healthy_shadow_rows(),
    )
    _write_csv(
        shadow_csv,
        [
            {key: value for key, value in row.items() if key != "frame_width"}
            for row in rows
        ],
    )

    with pytest.raises(ValueError, match="frame_width"):
        build_fixed_door_yaw_readiness(
            checkpoint,
            simulation,
            shadow,
            shadow_csv,
        )


def test_fixed_door_readiness_binds_action_contract_identity(tmp_path) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)
    summary = json.loads(shadow.read_text())
    summary["action_contract_sha256"] = "wrong"
    shadow.write_text(json.dumps(summary))

    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )

    assert report["next_live_gate_passed"] is False


def test_fixed_door_readiness_rejects_missing_action_contract(tmp_path) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)
    payload = json.loads(simulation.read_text())
    payload["trained_identity"].pop("action_contract")
    simulation.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="action contract"):
        build_fixed_door_yaw_readiness(
            checkpoint,
            simulation,
            shadow,
            shadow_csv,
        )


def test_fixed_door_readiness_rejects_missing_policy_contract(tmp_path) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)
    payload = json.loads(simulation.read_text())
    payload["trained_identity"].pop("policy_contract")
    simulation.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="policy contract"):
        build_fixed_door_yaw_readiness(
            checkpoint,
            simulation,
            shadow,
            shadow_csv,
        )


def test_fixed_door_readiness_loader_recomputes_csv_gate(tmp_path) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)
    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )
    rows = [
        row | {"phase": "track"}
        for row in _healthy_shadow_rows()
    ]
    _write_csv(
        shadow_csv,
        _bound_rows(checkpoint, simulation, rows),
    )
    report["shadow_csv_sha256"] = file_sha256(shadow_csv)
    readiness_path = tmp_path / "readiness.json"
    readiness = write_readiness(
        readiness_path,
        bind_fixed_door_readiness_identity(report, readiness_path),
    )

    with pytest.raises(ValueError, match="evidence no longer passes"):
        load_fixed_door_yaw_readiness(readiness, checkpoint, simulation)


def test_fixed_door_readiness_loader_rejects_mutated_shadow_summary(
    tmp_path,
) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)
    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )
    readiness_path = tmp_path / "readiness.json"
    readiness = write_readiness(
        readiness_path,
        bind_fixed_door_readiness_identity(report, readiness_path),
    )
    shadow.write_text(shadow.read_text() + "\n")

    with pytest.raises(ValueError, match="replay_report SHA-256"):
        load_fixed_door_yaw_readiness(readiness, checkpoint, simulation)


def test_fixed_door_readiness_loader_rejects_mutated_live_safety_contract(
    tmp_path,
) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)
    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )
    report["live_safety_contract"]["max_yawrate_deg_s"] = 9.0
    readiness_path = tmp_path / "readiness.json"
    readiness = write_readiness(
        readiness_path,
        bind_fixed_door_readiness_identity(report, readiness_path),
    )

    with pytest.raises(ValueError, match="live safety contract"):
        load_fixed_door_yaw_readiness(readiness, checkpoint, simulation)


def test_fixed_door_readiness_loader_rejects_mutated_evidence_age_contract(
    tmp_path,
) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)
    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )
    report["evidence_age_runtime_contract"]["maximum_evidence_age_s"] = 1.1
    readiness_path = tmp_path / "readiness.json"
    readiness = write_readiness(
        readiness_path,
        bind_fixed_door_readiness_identity(report, readiness_path),
    )

    with pytest.raises(ValueError, match="evidence-age runtime contract"):
        load_fixed_door_yaw_readiness(readiness, checkpoint, simulation)


def test_fixed_door_readiness_loader_rejects_limits_outside_live_contract(
    tmp_path,
) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)
    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )
    report["limits"]["detected_abs_yawrate_deg_s"] = 7.0
    readiness_path = tmp_path / "readiness.json"
    readiness = write_readiness(
        readiness_path,
        bind_fixed_door_readiness_identity(report, readiness_path),
    )

    with pytest.raises(ValueError, match="limits do not match"):
        load_fixed_door_yaw_readiness(readiness, checkpoint, simulation)


def test_fixed_door_readiness_loader_requires_exact_evaluation_path(
    tmp_path,
) -> None:
    checkpoint, simulation, shadow, shadow_csv = _evidence(tmp_path)
    report = build_fixed_door_yaw_readiness(
        checkpoint,
        simulation,
        shadow,
        shadow_csv,
    )
    readiness_path = tmp_path / "readiness.json"
    readiness = write_readiness(
        readiness_path,
        bind_fixed_door_readiness_identity(report, readiness_path),
    )
    copied = tmp_path / "copied-evaluation.json"
    copied.write_bytes(simulation.read_bytes())

    with pytest.raises(ValueError, match="evaluation report path"):
        load_fixed_door_yaw_readiness(readiness, checkpoint, copied)

    copied_readiness = tmp_path / "copied-readiness.json"
    copied_readiness.write_bytes(readiness.read_bytes())
    with pytest.raises(ValueError, match="readiness report path"):
        load_fixed_door_yaw_readiness(
            copied_readiness,
            checkpoint,
            simulation,
        )


def _evidence(tmp_path):
    checkpoint, simulation = write_test_promotion(tmp_path)
    return _shadow_artifacts(checkpoint, simulation, tmp_path)


def _shadow_artifacts(checkpoint, simulation, tmp_path, *, rows=None):
    simulation_payload = json.loads(simulation.read_text())
    evidence = validate_fixed_door_live_evidence(checkpoint, simulation)
    source_rows = _healthy_shadow_rows() if rows is None else rows
    rows = bind_fixed_door_shadow_rows(
        source_rows,
        _shadow_identity(evidence),
        evidence.bundle.action_contract,
    )
    dropped_frames = int(rows[0]["stream_dropped_frames"])
    shadow = tmp_path / "shadow.summary.json"
    shadow.write_text(
        json.dumps(
            summarize_shadow_rows(
                rows,
                checkpoint=checkpoint,
                training_report=simulation,
                simulation_gate=simulation_payload["simulation_gate"],
                dropped_frames=dropped_frames,
            )
        )
    )
    shadow_csv = tmp_path / "shadow.csv"
    _write_csv(shadow_csv, rows)
    return checkpoint, simulation, shadow, shadow_csv


def _shadow_identity(evidence):
    return build_fixed_door_shadow_identity(
        evidence,
        prompt=APPROVED_SHADOW_PROMPT,
        detector_model_id=APPROVED_SHADOW_DETECTOR_MODEL_ID,
        threshold=APPROVED_SHADOW_THRESHOLD,
        device=APPROVED_SHADOW_DEVICE,
        hardware_config=APPROVED_SHADOW_HARDWARE_CONFIG,
    )


def _bound_rows(checkpoint, simulation, rows):
    evidence = validate_fixed_door_live_evidence(checkpoint, simulation)
    return bind_fixed_door_shadow_rows(
        rows,
        _shadow_identity(evidence),
        evidence.bundle.action_contract,
    )


def _write_csv(path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _healthy_shadow_rows() -> list[dict]:
    left_detection = json.dumps(
        {"box": {"x_min": 0.0, "x_max": 0.2}}
    )
    return [
        {
            "frame_index": index,
            "frame_host_time_s": index / 10.0,
            "frame_width": 324,
            "frame_height": 244,
            "action_forward": 0.1,
            "action_yaw": 0.1,
            "controls_drone": False,
            "monitor_only": True,
            "phase": "search" if index < 20 else "track",
            "target_detected": index >= 20,
            "detection": None if index < 20 else left_detection,
            "inference_ms": 1.0,
            "grounding_age_s": None if index < 20 else 0.5,
            "grounding_inference_ms": 500.0,
            "grounding_result_frame_index": index,
            "stream_dropped_frames": 0,
        }
        for index in range(200)
    ]
