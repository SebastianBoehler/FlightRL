from __future__ import annotations

import hashlib
import json
from pathlib import Path


def write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value))
    return path


def valid_inputs(tmp_path: Path) -> dict[str, Path]:
    checkpoint = tmp_path / "policy.pt"
    bundle = tmp_path / "policy.edge-v3.bin"
    checkpoint.write_bytes(b"checkpoint")
    bundle.write_bytes(b"bundle")
    return {
        "audit": write_json(
            tmp_path / "audit.json",
            {"transfer_ready": True, "blocking_items": []},
        ),
        "profile": write_json(
            tmp_path / "profile.json",
            {"summary": {"profile_ready": True, "failures": []}},
        ),
        "config_export": write_json(
            tmp_path / "export.json",
            {"exported": True, "failures": []},
        ),
        "deployment_readiness": write_json(
            tmp_path / "readiness.json",
            {
                "schema": "flightrl.edge_v3.deployment_readiness.v1",
                "target": "ai_deck_gap8",
                "evidence_scope": "edge_deployment",
                "deployment_authority": True,
                "summary": {"total": 1, "ready": 1, "blocked": 0},
                "records": [
                    {
                        "task": "obstacle_avoidance",
                        "tasks": ["obstacle_avoidance"],
                        "controller": "policy",
                        "checkpoint": str(checkpoint),
                        "checkpoint_identity": identity(checkpoint),
                        "bundle": str(bundle),
                        "bundle_identity": identity(bundle),
                        "ready": True,
                        "failures": [],
                    }
                ],
            },
        ),
        "sim_readiness": write_json(
            tmp_path / "sim.json",
            {
                "evidence_scope": "desktop_development",
                "deployment_authority": False,
                "summary": {"total": 1, "ready": 1, "blocked": 0},
                "records": [
                    {
                        "task": "obstacle_avoidance",
                        "ready": True,
                        "failures": [],
                    }
                ],
            },
        ),
        "room_report": write_json(
            tmp_path / "room.json",
            {
                "summary": {
                    "mapping_ready": True,
                    "failures": [],
                    "point_count": 100,
                },
                "room_estimate": {
                    "width_m": 2.0,
                    "depth_m": 3.0,
                    "height_m": 2.5,
                },
            },
        ),
        "live_safety": write_json(
            tmp_path / "live-safety.json",
            {
                "summary": {
                    "passed": True,
                    "failures": [],
                    "checked": 2,
                    "hardware_scripts": 1,
                    "learned_checkpoint_hardware_scripts": 0,
                }
            },
        ),
    }


def identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
