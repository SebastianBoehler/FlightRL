import numpy as np
import pytest
from flightrl.inspection_fixture import three_panel_room
from flightrl.inspection_replay import capture_inspection
from flightrl.inspection_replay_io import (
    write_replay,
    load_replay,
    write_scene,
    load_scene,
)
from flightrl.scenario_replay import operator_frame


def capture(ticks=3):
    scene = three_panel_room()
    positions = np.array([[2.6, -1.5, 1.5], [1, 0, 1.5], [2.6, 1.5, 1.5]], np.float32)
    connected = np.ones((ticks + 1, 3), bool)
    connected[1:3, 0] = False
    meta, arrays = capture_inspection(
        scene, np.zeros((ticks, 3, 4), np.float32), positions, connected
    )
    return scene, meta, arrays


def test_capture_events_incomplete_and_round_trip(tmp_path):
    scene, meta, arrays = capture()
    assert meta["results"][0]["observed_inspected"] == ["red"]
    assert meta["results"][1]["evaluator_inspected_ids"] == []
    assert not any(r["evaluator_complete"] for r in meta["results"])
    assert operator_frame(arrays, 1, 0) is None
    assert operator_frame(arrays, 3, 0) is not None
    assert {e["type"] for e in meta["events"]} >= {
        "link_lost",
        "link_restored",
        "budget_exhausted",
        "inspected_observed",
    }
    write_scene(scene, tmp_path / "scene")
    restored = load_scene(tmp_path / "scene")
    assert restored.manifest == scene.manifest
    write_replay(meta, arrays, tmp_path / "replay")
    loaded, data = load_replay(tmp_path / "replay")
    assert loaded["events"] == meta["events"]
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], data[key])
    with pytest.raises(FileExistsError):
        write_replay(meta, arrays, tmp_path / "replay")
    (tmp_path / "replay" / "frames_local.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        load_replay(tmp_path / "replay")


def test_collision_is_terminal_and_records_attempt():
    scene = three_panel_room()
    ticks = 100
    actions = np.zeros((ticks, 1, 4), np.float32)
    actions[:, :, 2] = 0.5
    meta, arrays = capture_inspection(
        scene,
        actions,
        np.array([[1.8, 0, 1.5]], np.float32),
        np.ones((ticks + 1, 1), bool),
    )
    assert meta["ticks"] < ticks
    assert arrays["collisions_truth"][-1, 0]
    np.testing.assert_array_equal(
        arrays["states_truth"][-1], arrays["states_truth"][-2]
    )
    assert not np.array_equal(
        arrays["attempted_position_truth"][-1], arrays["states_truth"][-1, :, :3]
    )
    assert any(e["type"] == "collision" for e in meta["events"])


def test_scene_tampering_rejected(tmp_path):
    scene = three_panel_room()
    write_scene(scene, tmp_path / "scene")
    panels = np.load(tmp_path / "scene" / "panels.npy")
    panels[0, 11] = 210
    np.save(tmp_path / "scene" / "panels.npy", panels)
    with pytest.raises(ValueError, match="identity"):
        load_scene(tmp_path / "scene")


def test_environment_scene_round_trip_preserves_identity(tmp_path):
    from flightrl.inspection.industrial import utility_plant

    scene = utility_plant(400)
    write_scene(scene, tmp_path / "scene")
    restored = load_scene(tmp_path / "scene")
    assert restored.manifest == scene.manifest
    assert restored.environment == scene.environment
