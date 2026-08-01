from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from flightrl.puffer4_door_live_evidence import (
    validate_fixed_door_live_evidence,
)
from flightrl.puffer4_door_shadow_identity import (
    APPROVED_SHADOW_DETECTOR_MODEL_ID,
    APPROVED_SHADOW_DEVICE,
    APPROVED_SHADOW_HARDWARE_CONFIG,
    APPROVED_SHADOW_PROMPT,
    APPROVED_SHADOW_THRESHOLD,
    SHADOW_IDENTITY_JSON_FIELD,
    SHADOW_IDENTITY_SHA256_FIELD,
    build_fixed_door_shadow_identity,
    decode_fixed_door_shadow_identity,
    require_shadow_identity_matches_evidence,
)
from flightrl.puffer4_door_shadow_io import read_shadow_csv_evidence
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


def test_shadow_identity_binds_the_approved_runtime_and_evidence() -> None:
    evidence = validate_fixed_door_live_evidence(V59_CHECKPOINT, V59_REPORT)

    identity = build_fixed_door_shadow_identity(
        evidence,
        prompt=APPROVED_SHADOW_PROMPT,
        detector_model_id=APPROVED_SHADOW_DETECTOR_MODEL_ID,
        threshold=APPROVED_SHADOW_THRESHOLD,
        device=APPROVED_SHADOW_DEVICE,
        hardware_config=APPROVED_SHADOW_HARDWARE_CONFIG,
    )

    assert identity.payload["schema"] == "flightrl.fixed_door.real_shadow.v1"
    assert identity.payload["checkpoint"] == {
        "path": str(V59_CHECKPOINT.resolve()),
        "sha256": (
            "f676d12b9d37c27f4cc62f99beceec8f30e74c88be8564cb242c23755e202cce"
        ),
    }
    assert identity.payload["evaluation_report"] == {
        "path": str(V59_REPORT.resolve()),
        "sha256": (
            "b919e4f9951ad28904ce6cc7ee9b7a0f7b76ee70fba387e673bcda27a9bdcbbc"
        ),
    }
    assert identity.payload["action_contract"] == {
        "contract_id": "fixed-door-v59-legacy-physics-yaw-v1",
        "sha256": (
            "e666cf9c708b43e344355bad1c8b8f4c62826a79a26046839677e454879a80b5"
        ),
    }
    assert identity.payload["policy_contract"] == {
        "contract_id": "fixed-door-recurrent-policy-v1",
        "sha256": (
            "ad6fa58f50a1c0754d572643a9d7affe65f3e73d4d814c51030c733588ef8058"
        ),
    }
    assert identity.payload["evidence_age_contract"] == {
        "contract_id": "fixed-door-evidence-age-runtime-v1",
        "sha256": (
            "5223d41205907f53fe2d175d252ba78ab26717f446ae8461b50019184ebe1f48"
        ),
    }
    assert identity.payload["monitor_only"] is True
    assert identity.payload["controls_drone"] is False
    assert identity.payload["inference_device"] == "mps"
    assert identity.payload["hardware_config"] == {
        "path": str(APPROVED_SHADOW_HARDWARE_CONFIG),
        "sha256": (
            "7fe3f7c45b91982c5ef734d9ef9bcd8c392b4193db0f7bca697d777aef819e73"
        ),
    }
    assert decode_fixed_door_shadow_identity(
        identity.canonical_json,
        identity.sha256,
    ) == identity


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("prompt", "door"),
        ("detector_model_id", "other/model"),
        ("threshold", 0.26),
        ("device", "cpu"),
        ("hardware_config", Path("/tmp/other.toml")),
    ),
)
def test_shadow_identity_rejects_unapproved_detector_runtime(
    field: str,
    value: object,
) -> None:
    evidence = validate_fixed_door_live_evidence(V59_CHECKPOINT, V59_REPORT)
    arguments = {
        "prompt": APPROVED_SHADOW_PROMPT,
        "detector_model_id": APPROVED_SHADOW_DETECTOR_MODEL_ID,
        "threshold": APPROVED_SHADOW_THRESHOLD,
        "device": APPROVED_SHADOW_DEVICE,
        "hardware_config": APPROVED_SHADOW_HARDWARE_CONFIG,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match="approved shadow detector"):
        build_fixed_door_shadow_identity(evidence, **arguments)


def test_shadow_identity_rejects_self_consistent_checkpoint_relabel() -> None:
    evidence = validate_fixed_door_live_evidence(V59_CHECKPOINT, V59_REPORT)
    identity = build_fixed_door_shadow_identity(
        evidence,
        prompt=APPROVED_SHADOW_PROMPT,
        detector_model_id=APPROVED_SHADOW_DETECTOR_MODEL_ID,
        threshold=APPROVED_SHADOW_THRESHOLD,
        device=APPROVED_SHADOW_DEVICE,
        hardware_config=APPROVED_SHADOW_HARDWARE_CONFIG,
    )
    relabelled = dict(identity.payload)
    relabelled["checkpoint"] = {
        **identity.payload["checkpoint"],
        "path": str((ROOT / "other.bin").resolve()),
    }
    canonical = json.dumps(
        relabelled,
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    decoded = decode_fixed_door_shadow_identity(
        canonical,
        hashlib.sha256(canonical.encode()).hexdigest(),
    )

    with pytest.raises(ValueError, match="checkpoint"):
        require_shadow_identity_matches_evidence(decoded, evidence)


def test_shadow_csv_rejects_mixed_run_identities(tmp_path: Path) -> None:
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
        [_row(0), _row(1)],
        identity,
        evidence.bundle.action_contract,
    )
    changed = dict(identity.payload)
    changed["evaluation_report"] = {
        **identity.payload["evaluation_report"],
        "path": str((tmp_path / "copied.json").resolve()),
    }
    changed_json = json.dumps(changed, sort_keys=True, separators=(",", ":"))
    import hashlib

    rows[1][SHADOW_IDENTITY_JSON_FIELD] = changed_json
    rows[1][SHADOW_IDENTITY_SHA256_FIELD] = hashlib.sha256(
        changed_json.encode()
    ).hexdigest()
    path = tmp_path / "shadow.csv"
    _write_csv(path, rows)

    with pytest.raises(ValueError, match="mixed run identities"):
        read_shadow_csv_evidence(path)


def _row(index: int) -> dict:
    return {
        "frame_index": index,
        "frame_host_time_s": index / 10.0,
        "frame_width": 128,
        "frame_height": 96,
        "action_forward": 0.1,
        "action_yaw": 0.0,
        "controls_drone": False,
        "monitor_only": True,
        "phase": "search",
        "target_detected": False,
        "detection": None,
        "inference_ms": 1.0,
        "grounding_age_s": None,
        "grounding_inference_ms": 500.0,
        "grounding_result_frame_index": index,
        "stream_dropped_frames": 0,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
