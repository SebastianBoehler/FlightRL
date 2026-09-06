"""Generate one bounded CPU diagnostic; no mission or trained policy is executed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from flightrl.navigation.mission_spec import ResolvedMissionPlan
from flightrl.scenario_bundle import compile_scenario_bundle, write_scenario_bundle
from flightrl.scenario_replay import capture_scenario_replay, operator_frame
from flightrl.scenario_replay_io import load_scenario_replay, write_scenario_replay
from flightrl.sixdof.geometry import BoxRoom
from flightrl.sixdof.physics import SixDofPhysicsProfile
from flightrl.sixdof.sensor_model import SixDofSensorProfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = compile_scenario_bundle(
        vehicle=SixDofPhysicsProfile(), terrain=BoxRoom(),
        sensor=SixDofSensorProfile(),
        mission=ResolvedMissionPlan(source_text="diagnostic: no mission", steps=()),
    )
    scenario = write_scenario_bundle(bundle, args.output / "scenario")
    actions = np.zeros((100, 2, 4), dtype=np.float32)
    actions[:, 1, 3] = 0.15
    connected = np.ones((101, 2), dtype=np.bool_)
    connected[30:70, 1] = False
    capture_start = perf_counter()
    metadata, arrays = capture_scenario_replay(
        scenario, actions=actions, connected=connected,
        initial_position_m=np.array([[0,0,1],[0.2,0,1]], dtype=np.float32),
        dt_s=0.05, scene_seed=7,
    )
    capture_s = perf_counter() - capture_start
    root = write_scenario_replay(metadata, arrays, args.output / "replay")
    manifest, loaded = load_scenario_replay(root)
    assert operator_frame(loaded, 40, 1) is None
    assert operator_frame(loaded, 70, 1) is not None
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], loaded[name])
    # Portable image artifacts are actual native output, without a display renderer.
    for tick in (0, 40, 70, 100):
        frame = loaded["frames_local"][tick, 1]
        (args.output / f"local-frame-{tick:03d}.pgm").write_bytes(b"P5\n64 48\n255\n" + frame.tobytes())
    report = {
        "status": "diagnostic_replay_verified_not_inspection_demo",
        "replay_sha256": manifest["sha256"], "scenario_sha256": bundle.manifest["sha256"],
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "backend": "native_cpu",
        "environments": 2, "simulated_seconds_each": 5,
        "capture_wall_s": capture_s, "compile_capture_write_verify_wall_s": perf_counter()-start,
        "local_frames": 202, "operator_frames_withheld": int((~connected).sum()),
        "training_performed": False, "recovery_performed": False,
        "mission_success": None, "learner_exchange_measured": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
