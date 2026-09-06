"""Digest-checked storage for native diagnostic replays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flightrl.artifact_identity import bind_payload, require_bound_payload, sha256_file
from flightrl.scenario_bundle import FRAME_CONTRACT
from flightrl.scenario_replay import CAMERA, SCHEMA, STATE_FIELDS


ARRAY_DTYPES = {
    "time_s": "<f8", "states_truth": "<f4", "actions": "<f4",
    "frames_local": "|u1", "connected": "|b1",
}


def write_scenario_replay(metadata: dict, arrays: dict[str, np.ndarray], output: str | Path) -> Path:
    """Write to a new directory; the manifest is the last completion marker."""
    _validate(metadata, arrays)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    descriptors = {}
    for name in ARRAY_DTYPES:
        path = root / f"{name}.npy"
        np.save(path, arrays[name], allow_pickle=False)
        descriptors[name] = {"sha256": sha256_file(path)}
    manifest = bind_payload({**metadata, "arrays": descriptors})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return root


def load_scenario_replay(root: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    metadata = require_bound_payload(manifest, label="scenario replay")
    descriptors = metadata.pop("arrays")
    if set(descriptors) != set(ARRAY_DTYPES):
        raise ValueError("replay arrays do not match schema")
    arrays = {}
    for name in ARRAY_DTYPES:
        path = root / f"{name}.npy"
        if sha256_file(path) != descriptors[name]["sha256"]:
            raise ValueError(f"replay {name} SHA-256 does not match")
        arrays[name] = np.load(path, allow_pickle=False)
    _validate(metadata, arrays)
    for value in arrays.values():
        value.setflags(write=False)
    return manifest, arrays


def _validate(metadata: dict, arrays: dict[str, np.ndarray]) -> None:
    required = {
        "schema": SCHEMA, "authority": "simulation_only", "deployment_authority": False,
        "controller": "open_loop_action_array_diagnostic", "policy": None,
        "mission_execution": False, "core_abi": 1,
        "state_source": "simulator_truth_evaluator_only_no_estimator",
        "link_model": "scripted_observed_status_no_rf_physics",
        "clock": "simulation_seconds_frame_k_before_action_k",
        "state_fields": STATE_FIELDS, "camera": CAMERA,
        "frames": FRAME_CONTRACT,
        "action_fields": ["thrust", "roll_rate", "pitch_rate", "yaw_rate"],
        "action_units": "normalized_native_sixdof",
        "inspection_evaluation": "not_implemented",
    }
    for key, value in required.items():
        if key not in metadata or metadata[key] != value or type(metadata[key]) is not type(value):
            raise ValueError(f"unsupported replay {key}")
    for key in ("scenario_sha256", "native_binary_sha256", "recorder_sha256"):
        value = metadata.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid replay {key}")
    ticks, count = metadata.get("ticks"), metadata.get("num_envs")
    if type(ticks) is not int or type(count) is not int or not (1 <= ticks <= 1000 and 1 <= count <= 64):
        raise ValueError("invalid replay dimensions")
    dt = metadata.get("dt_s")
    if type(dt) not in (int, float) or not np.isfinite(dt) or not 0 < dt <= 0.05:
        raise ValueError("invalid replay dt_s")
    seed = metadata.get("scene_seed")
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("invalid replay scene_seed")
    shapes = {
        "time_s": (ticks + 1,), "states_truth": (ticks + 1, count, 14),
        "actions": (ticks, count, 4), "frames_local": (ticks + 1, count, 48, 64),
        "connected": (ticks + 1, count),
    }
    if set(arrays) != set(shapes):
        raise ValueError("replay arrays do not match schema")
    for name, shape in shapes.items():
        value = arrays[name]
        if value.shape != shape or value.dtype.str != ARRAY_DTYPES[name] or not value.flags.c_contiguous:
            raise ValueError(f"invalid replay {name} shape/dtype/layout")
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite replay {name}")
    if not np.array_equal(arrays["time_s"], np.arange(ticks + 1, dtype="<f8") * dt):
        raise ValueError("replay clock does not match dt_s")
    if np.any(np.abs(arrays["actions"]) > 1):
        raise ValueError("replay actions outside normalized bounds")
    if np.any(arrays["frames_local"] % 17):
        raise ValueError("replay frames violate gray4 contract")
