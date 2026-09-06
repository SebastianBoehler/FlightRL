import numpy as np
import pytest
from flightrl import _binding
from flightrl.inspection_fixture import three_panel_room
from flightrl.inspection_scene import (
    compile_inspection_scene,
    render_scene,
    swept_collision,
)
from flightrl.inspection_mission import InspectionMemory, detect_markers, evaluate_views


def pose(y=0, x=1):
    return np.array([[x, y, 1.5]], np.float32), np.array([[1, 0, 0, 0]], np.float32)


def test_visibility_occlusion_and_useful_view():
    scene = three_panel_room()
    p, q = pose(-1.5)
    frames, counts = render_scene(scene, p, q)
    assert counts[0, 0, 0] >= 64
    assert evaluate_views(scene, p, q, counts)[0, 0]
    p, q = pose()
    frames, counts = render_scene(scene, p, q)
    assert counts[0, 1, 1] > 0 and counts[0, 1, 0] == 0
    assert "green" not in {d["marker"] for d in detect_markers(frames[0])}
    q[:] = [0, 0, 0, 1]  # opposite direction, all panels behind camera
    _, counts = render_scene(scene, p, q)
    assert not counts.any()


def test_analytic_projected_rectangle_count_and_distance():
    scene = three_panel_room()
    p, q = pose(-1.5)
    _, counts = render_scene(scene, p, q)
    # Independent pinhole projection of red square, using pixel-center sampling.
    fy = np.tan(1.099557429 / 2)
    dx = 3.99 - (1 + 0.035)
    y = -(2 * (np.arange(64) + 0.5) / 64 - 1) * fy * 64 / 48 * dx
    z = -(2 * (np.arange(48) + 0.5) / 48 - 1) * fy * dx + 0.012
    expected = np.count_nonzero(abs(y) <= 0.45) * np.count_nonzero(abs(z) <= 0.45)
    assert counts[0, 0].tolist() == [expected, expected]
    far_p, _ = pose(-1.5, -1)
    _, far_counts = render_scene(scene, far_p, q)
    assert not evaluate_views(scene, far_p, q, far_counts)[0, 0]


def test_partial_occlusion_and_oblique_quality():
    scene = three_panel_room()
    # Place occluder edge across central panel image from this lateral viewpoint.
    p, q = pose(0.7)
    _, counts = render_scene(scene, p, q)
    assert 0 < counts[0, 1, 0] < counts[0, 1, 1]
    assert not evaluate_views(scene, p, q, counts)[0, 1]
    # Explicit oblique panel, front visible but cosine below frozen threshold.
    panels = scene.panels.copy()
    angle = np.deg2rad(60)
    panels[0, 0] = 3
    panels[0, 3:6] = [np.sin(angle), -np.cos(angle), 0]
    altered = compile_inspection_scene(scene.scenario, panels, scene.evaluator_ids)
    p, q = pose(-1.5, 1.5)
    _, counts = render_scene(altered, p, q)
    assert counts[0, 0, 0] > 0
    assert not evaluate_views(altered, p, q, counts)[0, 0]


def test_hidden_ids_do_not_change_pixels_or_discovery():
    scene = three_panel_room()
    p, q = pose(-1.5)
    changed = compile_inspection_scene(scene.scenario, scene.panels, (9, 8, 7))
    a, _ = render_scene(scene, p, q)
    b, _ = render_scene(changed, p, q)
    np.testing.assert_array_equal(a, b)
    assert detect_markers(a[0]) == detect_markers(b[0])
    removed = compile_inspection_scene(scene.scenario, scene.panels[1:], (102, 103))
    c, _ = render_scene(removed, p, q)
    assert "red" not in {d["marker"] for d in detect_markers(c[0])}
    assert detect_markers(np.zeros_like(a[0])) == []


def test_unique_memory_budget_and_no_completion_oracle():
    scene = three_panel_room()
    frame = render_scene(scene, *pose(-1.5, 2.6))[0][0]
    memory = InspectionMemory(3)
    for k in range(3):
        memory.observe(frame, k)
    assert memory.inspected == {"red"}
    assert memory.duplicate_views == 2
    assert memory.status == "budget_exhausted_coverage_unknown"
    assert sum(e["type"] == "inspected_observed" for e in memory.events) == 1
    with pytest.raises(ValueError):
        memory.observe(frame, 3)


def test_swept_collision_catches_tunneling_clearance_and_walls():
    scene = three_panel_room()
    start = np.array(
        [[1, 0, 1], [1, 1, 1], [1, 0.60, 1], [3.8, 2, 1], [2.1, 0, 1]], np.float32
    )
    end = np.array(
        [[3, 0, 1], [3, 1, 1], [3, 0.60, 1], [4.1, 2, 1], [2.1, 0, 1]], np.float32
    )
    assert swept_collision(scene, start, end).tolist() == [1, 0, 1, 1, 1]


def test_immutable_panels_and_native_shape_validation():
    scene = three_panel_room()
    with pytest.raises(ValueError):
        scene.panels.setflags(write=True)
    with pytest.raises(ValueError):
        _binding.inspection_render(
            np.array(1, np.float32), None, None, None, None, None, None
        )
    bad = scene.panels.copy()
    bad[0, 3] = 5
    with pytest.raises(ValueError, match="orthonormal"):
        compile_inspection_scene(scene.scenario, bad, scene.evaluator_ids)


def test_clipped_panel_is_not_a_useful_inspection():
    scene = three_panel_room()
    p, q = pose(-1.5, 3.5)
    _, counts = render_scene(scene, p, q)
    assert counts[0, 0, 0] >= 64
    assert not evaluate_views(scene, p, q, counts)[0, 0]
