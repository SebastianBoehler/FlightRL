import numpy as np
import pytest
from flightrl.inspection_fixture import three_panel_room
from flightrl.inspection_scene import compile_inspection_scene
from flightrl.inspection.rollout import run_mission
from flightrl.inspection.mapping import ObservedMap
from flightrl.inspection.metal import MetalCamera
from flightrl import _binding


def test_closed_loop_inspects_all_and_returns_on_observed_link_loss():
    scene = three_panel_room()
    result, records, frames, _, _ = run_mission(scene, ticks=650)
    assert result["coverage"] == 1 and not result["collision"]
    assert (
        np.linalg.norm(np.array(records[-1]["position"]) - records[0]["position"]) > 3
    )
    assert len(frames) == len(records)
    recovery, records, frames, _, _ = run_mission(scene, ticks=650, link_loss=True)
    assert recovery["recovered"] and not recovery["collision"]
    outage = [r for r in records if not r["connected"]]
    assert outage and records[-1]["connected"]
    assert records[-1]["position"][0] < 0.65
    assert len(frames) == len(records)


def test_actor_is_invariant_to_evaluator_ids():
    scene = three_panel_room()
    renamed = compile_inspection_scene(scene.scenario, scene.panels, (31, 51, 71))
    _, first, *_ = run_mission(scene, ticks=230)
    _, second, *_ = run_mission(renamed, ticks=230)
    np.testing.assert_array_equal(
        [r["command"] for r in first], [r["command"] for r in second]
    )
    np.testing.assert_array_equal(
        [r["position"] for r in first], [r["position"] for r in second]
    )


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("blocked_route", "recovery_blocked"),
        ("continued_outage", "outage_at_start"),
        ("estimator_loss", "localization_lost"),
    ],
)
def test_recovery_failure_envelopes(fault, expected):
    result, *_ = run_mission(three_panel_room(), ticks=900, link_loss=True, fault=fault)
    assert result["status"] == expected and not result["collision"]
    assert not result["recovered"]


def test_map_cannot_plan_through_unobserved_cells():
    mapping = ObservedMap()
    mapping.free.update([(0, 0), (1, 0), (3, 0), (4, 0)])
    assert mapping.path([0.125, 0.125], [1.125, 0.125]) == []


def test_metal_depth_and_rgb_match_native():
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("Apple Metal unavailable")
    scene = three_panel_room()
    p = np.array([[1, -1.5, 1.5], [1, 0, 1.5]], np.float32)
    q = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], np.float32)
    rgb = np.zeros((2, 48, 64, 3), np.uint8)
    depth = np.full((2, 48, 64), np.nan, np.float32)
    counts = np.zeros((2, 3, 2), np.int32)
    _binding.inspection_render(
        p,
        q,
        scene.scenario.arrays["terrain_bounds"],
        scene.scenario.arrays["terrain_obstacles"],
        scene.panels,
        rgb,
        counts,
        depth,
    )
    camera = MetalCamera(scene, 2)
    a, b = camera.render(p, q)
    torch.mps.synchronize()
    np.testing.assert_array_equal(a.cpu().numpy(), rgb)
    np.testing.assert_allclose(b.cpu().numpy(), depth, atol=2e-5, rtol=0)


def test_range_limit_is_free_space_without_an_invented_wall():
    mapping = ObservedMap()
    mapping.update(
        np.full((48, 64), 8.0, np.float32),
        np.array([0.0, 0, 1.0]),
        np.array([1.0, 0, 0, 0]),
    )
    assert len(mapping.free) > 1
    assert not mapping.occupied
