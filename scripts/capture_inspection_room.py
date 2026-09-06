"""Record three fixed-start native hover diagnostics, not autonomous exploration."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from flightrl.inspection_fixture import three_panel_room
from flightrl.inspection_replay import capture_inspection
from flightrl.inspection_replay_io import (
    write_scene,
    load_scene,
    write_replay,
    load_replay,
)
from flightrl.scenario_replay import operator_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    scene = three_panel_room()
    write_scene(scene, args.output / "scene")
    scene = load_scene(args.output / "scene")
    ticks = 100
    actions = np.zeros((ticks, 3, 4), np.float32)
    connected = np.ones((ticks + 1, 3), bool)
    connected[30:70] = False
    positions = np.array([[2.6, -1.5, 1.5], [1, 0, 1.5], [2.6, 1.5, 1.5]], np.float32)
    metadata, arrays = capture_inspection(scene, actions, positions, connected)
    manifest = write_replay(metadata, arrays, args.output / "replay")
    loaded, data = load_replay(args.output / "replay")
    assert loaded == manifest
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], data[name])
    assert operator_frame(data, 40, 0) is None
    assert operator_frame(data, 70, 0) is not None
    for env in range(3):
        frame = data["frames_local"][0, env]
        (args.output / f"camera-{env}.ppm").write_bytes(
            b"P6\n64 48\n255\n" + frame.tobytes()
        )
    inspected = set().union(
        *(set(r["evaluator_inspected_ids"]) for r in metadata["results"])
    )
    report = {
        "status": "inspection_geometry_and_memory_diagnostic_verified",
        "scene_sha256": scene.manifest["sha256"],
        "replay_sha256": manifest["sha256"],
        "backend": "native_cpu",
        "wall_s": perf_counter() - start,
        "local_frames": int(np.prod(data["connected"].shape)),
        "operator_frames_withheld": int((~data["connected"]).sum()),
        "diagnostic_viewpoint_union_inspected": sorted(inspected),
        "diagnostic_viewpoint_union_missed": sorted(
            set(scene.evaluator_ids) - inspected
        ),
        "results": metadata["results"],
        "training_performed": False,
        "navigation_performed": False,
        "recovery_performed": False,
        "localization": "not_implemented",
        "quality_scope": "ideal_instantaneous_camera_and_unique_solid_color_markers",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
