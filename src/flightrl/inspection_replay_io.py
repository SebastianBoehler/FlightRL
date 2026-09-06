"""Versioned inspection replay and scene storage, bound by content digests."""

import json
from pathlib import Path
import numpy as np

from flightrl.artifact_identity import bind_payload, require_bound_payload, sha256_file
from flightrl.scenario_bundle import write_scenario_bundle, load_scenario_bundle
from flightrl.inspection_scene import CAMERA, compile_inspection_scene
from flightrl.environment import EnvironmentProfile

DTYPES = {
    "time_s": "<f8",
    "states_truth": "<f4",
    "actions": "<f4",
    "frames_local": "|u1",
    "connected": "|b1",
    "panel_counts_truth": "<i4",
    "collisions_truth": "|u1",
    "attempted_position_truth": "<f4",
}


def write_scene(scene, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write_scenario_bundle(scene.scenario, root / "scenario")
    np.save(root / "panels.npy", scene.panels, allow_pickle=False)
    (root / "manifest.json").write_text(
        json.dumps(dict(scene.manifest), indent=2) + "\n"
    )


def load_scene(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    payload = require_bound_payload(manifest, label="inspection scene")
    scene = compile_inspection_scene(
        load_scenario_bundle(root / "scenario"),
        np.load(root / "panels.npy", allow_pickle=False),
        payload["evaluator_ids"],
        environment=EnvironmentProfile(**payload["environment"])
        if "environment" in payload
        else None,
    )
    if scene.manifest["sha256"] != manifest["sha256"]:
        raise ValueError("inspection scene contents do not match identity")
    return scene


def validate(metadata, arrays):
    if (
        metadata.get("schema") != "flightrl.inspection_replay.v2"
        or metadata.get("authority") != "simulation_only"
        or metadata.get("deployment_authority") is not False
        or metadata.get("policy") is not None
        or metadata.get("controller") != "open_loop_diagnostic"
        or metadata.get("camera") != CAMERA
    ):
        raise ValueError("unsupported inspection replay authority/schema/camera")
    for key in (
        "scenario_sha256",
        "scene_sha256",
        "native_binary_sha256",
        "recorder_sha256",
    ):
        digest = metadata.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid replay identity")
    if metadata.get("state_source") != "simulator_truth_evaluator_only_no_estimator":
        raise ValueError("unsupported state authority")
    t, n, p = (metadata.get(k) for k in ("ticks", "num_envs", "panel_count"))
    if any(type(x) is not int for x in (t, n, p)) or not (
        1 <= t <= 1000 and 1 <= n <= 64 and 0 <= p <= 1024
    ):
        raise ValueError("invalid replay dimensions")
    requested = metadata.get("requested_ticks")
    if (
        type(requested) is not int
        or not t <= requested <= 1000
        or len(metadata.get("results", [])) != n
    ):
        raise ValueError("invalid replay budget/results")
    dt = metadata.get("dt_s")
    if type(dt) not in (int, float) or not np.isfinite(dt) or not 0 < dt <= 0.05:
        raise ValueError("invalid replay clock")
    shapes = {
        "time_s": (t + 1,),
        "states_truth": (t + 1, n, 14),
        "actions": (t, n, 4),
        "frames_local": (t + 1, n, 48, 64, 3),
        "connected": (t + 1, n),
        "panel_counts_truth": (t + 1, n, p, 2),
        "collisions_truth": (t, n),
        "attempted_position_truth": (t, n, 3),
    }
    if set(arrays) != set(shapes):
        raise ValueError("unexpected replay arrays")
    for name, shape in shapes.items():
        a = arrays[name]
        if (
            a.shape != shape
            or a.dtype.str != DTYPES[name]
            or not a.flags.c_contiguous
            or not np.isfinite(a).all()
        ):
            raise ValueError(f"invalid replay array {name}")
    if not np.array_equal(arrays["time_s"], np.arange(t + 1, dtype=np.float64) * dt):
        raise ValueError("replay clock mismatch")
    if np.any(abs(arrays["actions"]) > 1) or np.any(arrays["collisions_truth"] > 1):
        raise ValueError("invalid actions or collision flags")
    counts = arrays["panel_counts_truth"]
    if (
        np.any(counts < 0)
        or np.any(counts > 3072)
        or np.any(counts[:, :, :, 0] > counts[:, :, :, 1])
    ):
        raise ValueError("invalid evaluator pixel counts")
    for event in metadata["events"]:
        if (
            type(event.get("tick")) is not int
            or not 0 <= event["tick"] <= t
            or type(event.get("env")) is not int
            or not 0 <= event["env"] < n
            or event.get("type")
            not in {
                "discovered",
                "inspected_observed",
                "budget_exhausted",
                "link_lost",
                "link_restored",
                "collision",
                "batch_stopped",
            }
        ):
            raise ValueError("invalid mission event")


def write_replay(metadata, arrays, root):
    validate(metadata, arrays)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    descriptors = {}
    for name in DTYPES:
        path = root / f"{name}.npy"
        np.save(path, arrays[name], allow_pickle=False)
        descriptors[name] = sha256_file(path)
    manifest = bind_payload({**metadata, "arrays": descriptors})
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def load_replay(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    metadata = require_bound_payload(manifest, label="inspection replay")
    descriptors = metadata.pop("arrays")
    if set(descriptors) != set(DTYPES):
        raise ValueError("unexpected replay arrays")
    arrays = {}
    for name in DTYPES:
        path = root / f"{name}.npy"
        if sha256_file(path) != descriptors[name]:
            raise ValueError(f"replay {name} digest mismatch")
        arrays[name] = np.load(path, allow_pickle=False)
    validate(metadata, arrays)
    for value in arrays.values():
        value.setflags(write=False)
    return manifest, arrays
