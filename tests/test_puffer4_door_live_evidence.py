from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from flightrl.puffer4_door_bundle import load_fixed_door_checkpoint_bundle
from flightrl.puffer4_door_contract import CORRECTED_DOOR_ACTION_CONTRACT
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_door_export import DOOR_NATIVE_FILES
from flightrl.puffer4_door_live_evidence import (
    V59_CHECKPOINT_SHA256,
    validate_fixed_door_live_evidence,
)
from flightrl.puffer4_door_policy_contract import door_policy_contract_report
from flightrl.puffer4_door_runtime import DoorPufferRuntime
from flightrl.puffer4_door_stream_contract import door_stream_contract_report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_lineage(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = (tmp_path / "door.bin").resolve()
    torch.save(DoorPufferRuntime(hidden_size=96).state_dict(), checkpoint)
    report = (tmp_path / "door.report.json").resolve()
    report.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "action_contract": CORRECTED_DOOR_ACTION_CONTRACT.to_report(),
                "policy_contract": door_policy_contract_report(
                    hidden_size=96,
                    num_layers=1,
                ),
                "procedural_stream_contract": door_stream_contract_report(),
                "config": {
                    "env_name": "flightrl_fixed_door_d1",
                    "seed": 23,
                },
            }
        )
    )
    return checkpoint, report


def _complete_run(*, camera: str = "full") -> dict:
    return {
        "status": "complete",
        "requested_steps": 10,
        "completed_steps": 10,
        "success_rate": 0.86 if camera == "full" else 0.01,
        "outside_fov_success_rate": 0.78,
        "collision_rate": 0.01,
        "condition": {
            "camera": camera,
            "recurrent_mode": "carried",
            "temporal_order": "ordered",
            "observation_challenge": False,
        },
        "finite_outputs": {
            "observations": True,
            "terminals": True,
            "policy_mean": True,
            "value": True,
            "recurrent_state": True,
            "actions": True,
            "metrics": True,
            "passed": True,
            "first_violation": None,
        },
    }


def _native_fingerprint(tmp_path: Path) -> dict:
    root = (tmp_path / "PufferLib").resolve()
    env_dir = root / "ocean" / "flightrl_fixed_door_d1"
    paths = (
        *(env_dir / name for name in ("binding.c", *DOOR_NATIVE_FILES)),
        root / "build.sh",
        root / "src/bindings_cpu.cpp",
        root / "src/vecenv.h",
        root / "src/tensor.h",
    )
    manifest = {
        str(path.resolve()): hashlib.sha256(str(path).encode()).hexdigest()
        for path in sorted(paths, key=lambda item: str(item.resolve()))
    }
    manifest_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    extension = root / "pufferlib" / "_C.test-extension"
    return {
        "schema_version": 1,
        "env_name": "flightrl_fixed_door_d1",
        "build_mode": "cpu",
        "python_abi": {
            "ext_suffix": ".test-extension",
            "cache_tag": "cpython-test",
        },
        "source_files_sha256": manifest,
        "source_manifest_sha256": manifest_digest,
        "source_manifest_sha256_before": manifest_digest,
        "source_manifest_sha256_after": manifest_digest,
        "extension": {
            "path": str(extension.resolve()),
            "sha256": hashlib.sha256(b"extension").hexdigest(),
        },
    }


def _write_promotion(
    tmp_path: Path,
    *,
    checkpoint: Path | None = None,
    lineage: Path | None = None,
) -> tuple[Path, Path]:
    if checkpoint is None or lineage is None:
        checkpoint, lineage = _write_lineage(tmp_path)
    bundle = load_fixed_door_checkpoint_bundle(checkpoint, lineage)
    output = (tmp_path / "door.promotion-evaluation.json").resolve()
    full = _complete_run()
    masked = _complete_run(camera="masked")
    reset = _complete_run()
    reset["condition"]["recurrent_mode"] = "reset_each_step"
    temporal = _complete_run()
    temporal["condition"]["temporal_order"] = "scrambled"
    live_cap = _complete_run()
    policy_yaw_scale = bundle.action_contract.max_yawrate_deg_s
    normalized_live_cap = 8.0 / policy_yaw_scale
    live_cap["yaw_cap"] = {
        "enabled": True,
        "normalized_limit": normalized_live_cap,
        "saturation_fraction": 0.2,
    }
    report = {
        "evaluation_schema": "flightrl.fixed_door.promotion.v3",
        "trained_identity": bundle.trained_identity(),
        "evaluation_identity": {
            "kind": "fixed_door_promotion",
            "schema_version": 1,
            "report": {"path": str(output)},
            "environment": {
                "name": bundle.env_name,
                "agents": 8,
                "steps_per_condition": 10,
                "seed": 10_011,
            },
            "action_contract_sha256": bundle.action_contract.sha256(),
            "policy_contract_sha256": bundle.policy_contract["sha256"],
            "procedural_stream_contract": door_stream_contract_report(),
            "native_build_fingerprint": _native_fingerprint(tmp_path),
            "evidence_age_runtime_contract": (
                FIXED_DOOR_EVIDENCE_AGE_CONTRACT.to_report()
            ),
        },
        "lineage": bundle.lineage(),
        "full_camera": full,
        "masked_camera": masked,
        "recurrence_reset_ablation": {"metrics": reset},
        "temporal_order_ablation": {"metrics": temporal},
        "simulation_gate": {
            "passed": True,
            "checks": {
                "completion": True,
                "collision": True,
                "outside_fov_completion": True,
                "camera_mask": True,
            },
            "failures": [],
        },
        "promotion_evidence": {
            "all_default_runs_complete": True,
            "all_default_outputs_finite": True,
            "temporal_order_run_complete": True,
            "temporal_order_outputs_finite": True,
        },
        "live_yaw_cap_challenge": {
            "label": "live_yaw_cap_only",
            "policy_max_yawrate_deg_s": policy_yaw_scale,
            "live_max_yawrate_deg_s": 8.0,
            "normalized_yaw_limit": normalized_live_cap,
            "condition": {
                "camera": "full",
                "recurrent_mode": "carried",
                "forward_action": "unchanged",
                "combined_with_other_ablation": False,
            },
            "metrics": live_cap,
        },
    }
    output.write_text(json.dumps(report))
    return checkpoint, output


def test_validator_accepts_complete_nested_promotion_v3(tmp_path: Path) -> None:
    checkpoint, report = _write_promotion(tmp_path)

    evidence = validate_fixed_door_live_evidence(checkpoint, report)

    assert evidence.kind == "promotion_v3"
    assert evidence.bundle.report_path == report


@pytest.mark.parametrize(
    ("field", "match"),
    (
        ("evaluation_schema", "promotion.v3"),
        ("lineage", "lineage"),
        ("native_build_fingerprint", "native build fingerprint"),
        ("native_source_digest", "manifest digest"),
        ("all_default_runs_complete", "incomplete"),
        ("live_yaw_cap_challenge", "live yaw-cap"),
        ("action_contract_sha256", "action contract"),
    ),
)
def test_validator_rejects_incomplete_or_relabelled_promotion(
    tmp_path: Path,
    field: str,
    match: str,
) -> None:
    checkpoint, report = _write_promotion(tmp_path)
    payload = json.loads(report.read_text())
    if field == "lineage":
        payload.pop(field)
    elif field == "native_build_fingerprint":
        payload["evaluation_identity"].pop(field)
    elif field == "native_source_digest":
        fingerprint = payload["evaluation_identity"]["native_build_fingerprint"]
        first_path = next(iter(fingerprint["source_files_sha256"]))
        fingerprint["source_files_sha256"][first_path] = "0" * 64
    elif field == "all_default_runs_complete":
        payload["promotion_evidence"][field] = False
    elif field == "live_yaw_cap_challenge":
        payload.pop(field)
    elif field == "action_contract_sha256":
        payload["evaluation_identity"][field] = "0" * 64
    else:
        payload[field] = "legacy"
    report.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=match):
        validate_fixed_door_live_evidence(checkpoint, report)


def test_validator_rejects_flat_non_v59_training_report(tmp_path: Path) -> None:
    checkpoint, training_report = _write_lineage(tmp_path)

    with pytest.raises(ValueError, match="promotion.v3"):
        validate_fixed_door_live_evidence(checkpoint, training_report)


def test_validator_grandfathers_only_exact_v59_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoint = (
        root
        / "artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m"
        / "flightrl_fixed_door_d1_seed11_1048576.bin"
    )
    report = checkpoint.with_suffix(".reevaluation.json")

    evidence = validate_fixed_door_live_evidence(checkpoint, report)

    assert evidence.kind == "grandfathered_v59"
    assert evidence.bundle.checkpoint_sha256 == V59_CHECKPOINT_SHA256


def test_exact_v59_can_gain_new_promotion_v3_evidence(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoint = (
        root
        / "artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m"
        / "flightrl_fixed_door_d1_seed11_1048576.bin"
    )
    lineage = checkpoint.with_suffix(".reevaluation.json")
    _, promotion = _write_promotion(
        tmp_path,
        checkpoint=checkpoint,
        lineage=lineage,
    )

    evidence = validate_fixed_door_live_evidence(checkpoint, promotion)

    assert evidence.kind == "promotion_v3"
