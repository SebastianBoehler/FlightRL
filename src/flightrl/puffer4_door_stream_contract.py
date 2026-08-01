from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def door_stream_contract_report() -> dict[str, Any]:
    payload = _payload()
    return payload | {"sha256": _sha256(payload)}


def verify_door_stream_contract(report: Mapping[str, Any]) -> None:
    payload = {key: value for key, value in report.items() if key != "sha256"}
    if report.get("sha256") != _sha256(payload):
        raise ValueError("fixed-door stream contract SHA-256 does not match")
    if payload != _payload():
        raise ValueError("fixed-door stream contract does not match runtime")


def _payload() -> dict[str, Any]:
    return {
        "contract_id": "fixed-door-episode-stream-v1",
        "schema_version": 1,
        "seed_derivation": "domain_separated_avalanche32",
        "physical_seed_inputs": [
            "base_seed",
            "environment_index",
            "episode_index_u64",
        ],
        "appearance_seed_inputs": [
            "base_seed",
            "appearance_seed",
            "environment_index",
            "episode_index_u64",
        ],
        "reset_point": "before_domain_scene_lighting_detector_draws",
        "episode_identity": "per_environment_nth_reset",
        "prior_episode_draw_count_affects_next_episode": False,
        "group_schema": {
            "version": 1,
            "kind": "marginal_not_joint",
            "scene_group_id_bits": {
                "0-1": "layout_family_0_to_3",
                "2-3": "door_face_0_to_3",
                "4": "low_light",
                "5": "obstacle_present",
                "6": "initial_outside_fov",
                "7": "reserved",
            },
            "category_zero": "derived_from_total_minus_positive_categories",
            "zero_support_success": None,
        },
    }


def _sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
