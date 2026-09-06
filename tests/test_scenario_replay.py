from pathlib import Path

import numpy as np
import pytest

from flightrl import _binding
from flightrl.artifact_identity import bind_payload
from flightrl.navigation.mission_spec import ResolvedMissionPlan
from flightrl.scenario_bundle import compile_scenario_bundle, write_scenario_bundle
from flightrl.scenario_replay import capture_scenario_replay, operator_frame
from flightrl.scenario_replay_io import load_scenario_replay, write_scenario_replay
from flightrl.sixdof.geometry import AxisAlignedObstacle, BoxRoom
from flightrl.sixdof.physics import SixDofPhysicsProfile
from flightrl.sixdof.sensor_model import SixDofSensorProfile


def scenario(tmp_path, *, terrain=None, sensor=None):
    bundle = compile_scenario_bundle(
        vehicle=SixDofPhysicsProfile(), terrain=terrain or BoxRoom(),
        sensor=sensor or SixDofSensorProfile(),
        mission=ResolvedMissionPlan(source_text="diagnostic: no mission", steps=()),
    )
    return write_scenario_bundle(bundle, tmp_path / "scenario"), bundle


def inputs():
    actions = np.zeros((6, 2, 4), dtype=np.float32)
    actions[:, 1, 3] = 0.4
    connected = np.ones((7, 2), dtype=np.bool_)
    connected[2:5, 1] = False
    return dict(actions=actions, connected=connected, dt_s=0.05, scene_seed=7,
                initial_position_m=np.array([[0, 0, 1], [0.1, 0, 1]], dtype=np.float32))


def test_native_capture_time_alignment_and_actual_frames(tmp_path):
    path, bundle = scenario(tmp_path)
    metadata, arrays = capture_scenario_replay(path, **inputs())
    assert metadata["scenario_sha256"] == bundle.manifest["sha256"]
    assert metadata["policy"] is None and metadata["mission_execution"] is False
    assert metadata["inspection_evaluation"] == "not_implemented"
    assert np.any(arrays["states_truth"][-1, 1, 6:10] != arrays["states_truth"][0, 1, 6:10])
    # Independent invocation proves frames were rendered from the stored pose at k,
    # not the previous pose or an unrelated preview stream.
    states = arrays["states_truth"][3]
    frame = np.empty((2, 48, 64), dtype=np.uint8)
    _binding.sixdof_render_gray4(
        np.ascontiguousarray(states[:, :3]), np.ascontiguousarray(states[:, 6:10]),
        bundle.arrays["terrain_bounds"], np.full(2, 60, np.float32),
        np.full(2, 7, np.int32), frame,
    )
    np.testing.assert_array_equal(frame, arrays["frames_local"][3])
    np.testing.assert_array_equal(arrays["time_s"], np.arange(7) * 0.05)
    # Action 0 must map state 0 to state 1, including thrust and body rates.
    state = arrays["states_truth"][0]
    p, v, q, w = [np.ascontiguousarray(state[:, a:b]) for a, b in ((0,3),(3,6),(6,10),(10,13))]
    thrust = state[:, 13].copy()
    _binding.sixdof_step(p, v, q, w, np.empty((2,6),np.float32), thrust,
                        arrays["actions"][0],
                        np.repeat(bundle.arrays["vehicle_physics"][None], 2, axis=0),
                        bundle.arrays["terrain_bounds"], 0.05)
    np.testing.assert_array_equal(np.column_stack((p,v,q,w,thrust)), arrays["states_truth"][1])


def test_dropout_retains_local_frames_but_operator_has_no_new_frame(tmp_path):
    path, _ = scenario(tmp_path)
    _, arrays = capture_scenario_replay(path, **inputs())
    assert operator_frame(arrays, 3, 1) is None
    assert arrays["frames_local"][3, 1].any()
    np.testing.assert_array_equal(operator_frame(arrays, 5, 1), arrays["frames_local"][5, 1])
    assert operator_frame(arrays, 3, 0) is not None
    with pytest.raises(IndexError):
        operator_frame(arrays, -1, 0)


def test_disk_roundtrip_repeatability_and_tampering(tmp_path):
    path, _ = scenario(tmp_path)
    first, arrays = capture_scenario_replay(path, **inputs())
    second, repeated = capture_scenario_replay(path, **inputs())
    assert first == second
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], repeated[key])
    root = write_scenario_replay(first, arrays, tmp_path / "replay")
    manifest, loaded = load_scenario_replay(root)
    assert manifest["scenario_sha256"] == first["scenario_sha256"]
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], loaded[key])
        assert not loaded[key].flags.writeable
    with pytest.raises(FileExistsError):
        write_scenario_replay(first, arrays, root)
    altered = loaded["frames_local"].copy()
    altered[0,0,0,0] ^= 17
    np.save(root / "frames_local.npy", altered)
    with pytest.raises(ValueError, match="SHA-256"):
        load_scenario_replay(root)


@pytest.mark.parametrize("change", ["nonfinite", "action_range", "link_shape", "dt", "outside"])
def test_rejects_invalid_runtime_inputs(tmp_path, change):
    path, _ = scenario(tmp_path)
    args = inputs()
    if change == "nonfinite": args["actions"][0,0,0] = np.nan
    if change == "action_range": args["actions"][0,0,0] = 2
    if change == "link_shape": args["connected"] = args["connected"][:-1]
    if change == "dt": args["dt_s"] = 0
    if change == "outside": args["initial_position_m"][0,0] = 9
    with pytest.raises(ValueError):
        capture_scenario_replay(path, **args)


def test_rejects_unsupported_geometry_and_sensor(tmp_path):
    obstacle = AxisAlignedObstacle(x_min=0.2,x_max=0.4,y_min=0.2,y_max=0.4,z_min=0,z_max=1)
    path, _ = scenario(tmp_path / "a", terrain=BoxRoom(obstacles=(obstacle,)))
    with pytest.raises(ValueError, match="obstacles"):
        capture_scenario_replay(path, **inputs())
    path, _ = scenario(tmp_path / "b", sensor=SixDofSensorProfile(state_noise_std_m=0.1))
    with pytest.raises(ValueError, match="ideal"):
        capture_scenario_replay(path, **inputs())


def test_rejects_resigned_false_authority_and_clock(tmp_path):
    import json
    path, _ = scenario(tmp_path)
    metadata, arrays = capture_scenario_replay(path, **inputs())
    arrays["time_s"][2] = 0
    with pytest.raises(ValueError, match="clock"):
        write_scenario_replay(metadata, arrays, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    arrays["time_s"] = np.arange(7) * 0.05
    root = write_scenario_replay(metadata, arrays, tmp_path / "replay")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.pop("sha256")
    manifest["deployment_authority"] = True
    (root / "manifest.json").write_text(json.dumps(bind_payload(manifest)))
    with pytest.raises(ValueError, match="authority"):
        load_scenario_replay(root)


def test_rejects_mission_rows_instead_of_ignoring_them(tmp_path):
    from flightrl.navigation.mission_spec import MissionCommand, ResolvedMissionStep, TargetAnchor
    mission = ResolvedMissionPlan(source_text="hold", steps=(ResolvedMissionStep(
        command=MissionCommand.HOLD, target_index=-1, anchor=TargetAnchor.PREFERRED,
        target_xyz_m=(0,0,1), target_yaw_rad=0, duration_s=1, speed_scale=1,
    ),))
    bundle = compile_scenario_bundle(vehicle=SixDofPhysicsProfile(), terrain=BoxRoom(),
                                     sensor=SixDofSensorProfile(), mission=mission)
    path = write_scenario_bundle(bundle, tmp_path / "scenario")
    with pytest.raises(ValueError, match="mission rows"):
        capture_scenario_replay(path, **inputs())


def test_rejects_trajectory_leaving_room(tmp_path):
    path, _ = scenario(tmp_path)
    args = inputs()
    args["initial_position_m"][:, 2] = 2.49
    args["actions"][:, :, 0] = 1
    with pytest.raises(ValueError, match="room interior"):
        capture_scenario_replay(path, **args)
