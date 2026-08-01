from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from flightrl.puffer4_door_bundle import load_fixed_door_checkpoint_bundle
from flightrl.puffer4_door_contract import CORRECTED_DOOR_ACTION_CONTRACT
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_door_export import DOOR_NATIVE_FILES
from flightrl.puffer4_door_policy_contract import door_policy_contract_report
from flightrl.puffer4_door_runtime import DoorPufferRuntime
from flightrl.puffer4_door_stream_contract import door_stream_contract_report


def write_test_lineage(tmp_path: Path) -> tuple[Path, Path]:
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


def write_test_promotion(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint, lineage = write_test_lineage(tmp_path)
    bundle = load_fixed_door_checkpoint_bundle(checkpoint, lineage)
    output = (tmp_path / "door.promotion-evaluation.json").resolve()
    full = _complete_run()
    masked = _complete_run(camera="masked")
    reset = _complete_run()
    reset["condition"]["recurrent_mode"] = "reset_each_step"
    temporal = _complete_run()
    temporal["condition"]["temporal_order"] = "scrambled"
    live_cap = _complete_run()
    live_cap["yaw_cap"] = {
        "enabled": True,
        "normalized_limit": 8.0 / 70.0,
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
            "procedural_stream_contract": bundle.stream_contract,
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
            "policy_max_yawrate_deg_s": 70.0,
            "live_max_yawrate_deg_s": 8.0,
            "normalized_yaw_limit": 8.0 / 70.0,
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
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "env_name": "flightrl_fixed_door_d1",
        "build_mode": "cpu",
        "python_abi": {
            "ext_suffix": ".test-extension",
            "cache_tag": "cpython-test",
        },
        "source_files_sha256": manifest,
        "source_manifest_sha256": digest,
        "source_manifest_sha256_before": digest,
        "source_manifest_sha256_after": digest,
        "extension": {
            "path": str(
                (root / "pufferlib" / "_C.test-extension").resolve()
            ),
            "sha256": hashlib.sha256(b"extension").hexdigest(),
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
