import numpy as np
import pytest
from flightrl.reconstruction.geometry import intrinsics, unproject
from flightrl.reconstruction.fusion import SurfaceMap
from flightrl.reconstruction.metrics import alignment
from flightrl.reconstruction.odometry import VisualOdometry
from flightrl.reconstruction.registration import register


def test_rgbd_owns_depth_when_capture_buffer_is_reused():
    depth = np.ones((192, 256), np.float32)
    tracker = VisualOdometry(intrinsics(), "rgbd")
    tracker.step(np.zeros((192, 256, 3), np.uint8), depth)
    depth[:] = 7
    np.testing.assert_array_equal(tracker.depth, 1)


def test_unobservable_monocular_scale_reports_unavailable():
    from flightrl.reconstruction.metrics import score

    result = score([np.eye(4)] * 3, np.array([np.eye(4)] * 3), [], [], "rgb")
    assert result["tracked_frames"] == 3
    assert result["ate_rmse_m"] is None
    assert result["surface_coverage"] is None
    assert result["unavailable_reason"] == "degenerate monocular scale"
    moving = np.array([np.eye(4)] * 3)
    moving[:, 0, 3] = [0, 1, 2]
    result = score(moving, np.array([np.eye(4)] * 3), [], [], "rgb")
    assert result["ate_rmse_m"] is None
    with pytest.raises(ValueError, match="Degenerate"):
        alignment(moving[:, :3, 3], np.zeros((3, 3)), True)


def test_axial_depth_uses_camera_to_world_without_inversion():
    from flightrl.reconstruction.geometry import axial_depth_points

    pose = np.eye(4)
    pose[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    pose[:3, 3] = [10, 20, 30]
    points = axial_depth_points(np.full((2, 2), 2.0), np.eye(3), pose)
    np.testing.assert_allclose(points[0, 0], [12, 20, 30])
    np.testing.assert_allclose(points[1, 1], [12, 22, 32])
    # Off-axis Z depth must not be normalized into ray distance.
    np.testing.assert_allclose(np.linalg.norm(points[1, 1] - pose[:3, 3]), np.sqrt(12))


def test_ray_distance_and_optical_projection():
    k = intrinsics()
    xy = np.array([[0.0, 0.0], [127.5, 95.5], [255.0, 191.0]])
    xyz = unproject(xy, np.array([3.0, 4.0, 5.0]), k)
    np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), [3, 4, 5])
    projected = xyz @ k.T
    np.testing.assert_allclose(projected[:, :2] / projected[:, 2, None], xy, atol=1e-10)


def test_empty_map_first_observation_not_overwritten():
    m = SurfaceMap()
    assert m.export() == []
    m.add([[0, 0, 1]], [[255, 0, 0]], 7)
    m.add([[0, 0, 1.01]], [[0, 255, 0]], 9)
    assert m.export() == [([0, 0, 1], [255, 0, 0], 7)]


def test_monocular_scale_alignment_is_evaluator_only():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(20, 3))
    r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    b = 2 * a @ r.T + [3, 4, 5]
    scale, rotation, t = alignment(a, b, True)
    np.testing.assert_allclose(scale * a @ rotation.T + t, b, atol=1e-10)


def test_blank_camera_does_not_invent_tracking_or_registration():
    image = np.zeros((192, 256, 3), np.uint8)
    k = intrinsics()
    mono = VisualOdometry(k, "rgb")
    assert mono.step(image) is None
    assert mono.step(image) is None
    assert mono.status == "lost"
    assert register(image, np.ones((192, 256)), image, k) == (None, 0)
    with pytest.raises(ValueError):
        VisualOdometry(k, "rgbd").step(image)


def test_monocular_rejects_depth_and_degenerate_scale():
    with pytest.raises(ValueError, match="must not receive depth"):
        VisualOdometry(intrinsics(), "rgb").step(
            np.zeros((192, 256, 3), np.uint8), np.ones((192, 256))
        )
    with pytest.raises(ValueError, match="Degenerate"):
        alignment(np.zeros((3, 3)), np.ones((3, 3)), True)


def test_native_ray_geometry_matches_evaluator_camera_frame():
    from flightrl import _binding
    from flightrl.reconstruction.experiment import camera_pose
    from flightrl.reconstruction.geometry import transform

    rgb = np.empty((1, 48, 64, 3), np.uint8)
    depth = np.empty((1, 48, 64), np.float32)
    p = np.array([[0, 0, 1]], np.float32)
    q = np.array([[1, 0, 0, 0]], np.float32)
    _binding.inspection_render(
        p,
        q,
        np.array([-3, 6, -4, 4, 0, 4, 8], np.float32),
        np.empty((0, 6), np.float32),
        np.empty((0, 14), np.float32),
        rgb,
        np.empty((1, 0, 2), np.int32),
        depth,
        0,
    )
    point = unproject(np.array([[31, 23]]), depth[0, 23, 31:32], intrinsics(64, 48))
    offset = (6 - 0.035) * np.tan(1.099557429 / 2) / 48
    np.testing.assert_allclose(
        transform(point, camera_pose(p[0], q[0])),
        [[6, offset, 1.012 + offset]],
        atol=1e-5,
    )
