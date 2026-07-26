from __future__ import annotations

import numpy as np

from flightrl import _binding


HEIGHT = 48
WIDTH = 64
PIXELS = HEIGHT * WIDTH
OBS_DIM = 3 * PIXELS + 6
ROOM = np.array([-2.0, 2.0, -2.0, 2.0, 0.0, 2.5, 4.0], dtype=np.float32)
PHYSICS = np.array(
    [0.036, 9.81, 0.10, 0.045, 0.75, 6.0, 6.0, 4.0, 0.0],
    dtype=np.float32,
)


def _pose_batch(count: int) -> tuple[np.ndarray, np.ndarray]:
    position = np.zeros((count, 3), dtype=np.float32)
    position[:, 2] = 0.65
    quaternion = np.zeros((count, 4), dtype=np.float32)
    quaternion[:, 0] = 1.0
    return position, quaternion


def _render(position: np.ndarray, quaternion: np.ndarray, target_mean: float = 60.0) -> np.ndarray:
    count = position.shape[0]
    frames = np.empty((count, HEIGHT, WIDTH), dtype=np.uint8)
    means = np.full(count, target_mean, dtype=np.float32)
    seeds = np.full(count, 7, dtype=np.int32)
    _binding.sixdof_render_gray4(position, quaternion, ROOM, means, seeds, frames)
    return frames


def test_native_camera_matches_gray4_wire_contract() -> None:
    position, quaternion = _pose_batch(2)
    frames = _render(position, quaternion, target_mean=70.0)

    assert frames.shape == (2, HEIGHT, WIDTH)
    assert np.all(frames % 17 == 0)
    assert abs(float(frames.mean()) - 70.0) < 8.0
    assert np.unique(frames).size >= 4


def test_native_camera_changes_with_translation_and_yaw() -> None:
    position, quaternion = _pose_batch(3)
    position[1, :2] = (0.45, -0.25)
    half_yaw = np.deg2rad(45.0)
    quaternion[2] = (np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw))

    frames = _render(position, quaternion)

    assert np.mean(frames[0] != frames[1]) > 0.10
    assert np.mean(frames[0] != frames[2]) > 0.25


def test_native_visual_observation_has_temporal_and_intent_channels() -> None:
    position, quaternion = _pose_batch(1)
    target = np.array([[1.0, 0.0, 0.65]], dtype=np.float32)
    target_yaw = np.zeros(1, dtype=np.float32)
    means = np.array([60.0], dtype=np.float32)
    seeds = np.array([11], dtype=np.int32)
    previous = np.zeros((1, HEIGHT, WIDTH), dtype=np.uint8)
    reset = np.ones(1, dtype=np.uint8)
    observation = np.empty((1, OBS_DIM), dtype=np.float32)

    _binding.sixdof_visual_observation(
        position,
        quaternion,
        target,
        target_yaw,
        ROOM,
        means,
        seeds,
        previous,
        reset,
        observation,
    )

    assert np.max(np.abs(observation[:, PIXELS : 2 * PIXELS])) == 0.0
    assert np.max(observation[:, 2 * PIXELS : 3 * PIXELS]) == 0.0
    np.testing.assert_allclose(observation[0, -6:], [1.0, 0.0, 0.0, 0.25, 0.0, 1.0], atol=1e-6)

    position[0, 1] = 0.3
    reset[0] = 0
    _binding.sixdof_visual_observation(
        position,
        quaternion,
        target,
        target_yaw,
        ROOM,
        means,
        seeds,
        previous,
        reset,
        observation,
    )

    assert np.max(np.abs(observation[:, PIXELS : 2 * PIXELS])) > 0.08
    assert np.sum(observation[:, 2 * PIXELS : 3 * PIXELS]) > 0


def test_native_intent_is_expressed_in_body_frame() -> None:
    position, quaternion = _pose_batch(1)
    half_yaw = np.deg2rad(45.0)
    quaternion[0] = (np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw))
    target = np.array([[1.0, 0.0, 0.65]], dtype=np.float32)
    target_yaw = np.zeros(1, dtype=np.float32)
    previous = np.zeros((1, HEIGHT, WIDTH), dtype=np.uint8)
    observation = np.empty((1, OBS_DIM), dtype=np.float32)

    _binding.sixdof_visual_observation(
        position,
        quaternion,
        target,
        target_yaw,
        ROOM,
        np.array([60.0], dtype=np.float32),
        np.array([3], dtype=np.int32),
        previous,
        np.ones(1, dtype=np.uint8),
        observation,
    )

    np.testing.assert_allclose(observation[0, -6:], [0.0, -1.0, 0.0, 0.25, -1.0, 0.0], atol=1e-5)


def test_navigation_setpoints_use_firmware_style_inner_control() -> None:
    velocity = np.zeros((2, 3), dtype=np.float32)
    quaternion = np.zeros((2, 4), dtype=np.float32)
    quaternion[:, 0] = 1.0
    setpoints = np.array([[1.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    output = np.empty((2, 4), dtype=np.float32)

    _binding.sixdof_setpoint_actions(
        velocity,
        quaternion,
        setpoints,
        np.repeat(PHYSICS[None, :], 2, axis=0),
        output,
        0.2,
        0.1,
        3.0,
        6.0,
        2.0,
    )

    assert output[0, 0] > 0.0
    assert output[0, 2] > 0.0
    assert output[0, 3] == 0.5
    np.testing.assert_allclose(output[1], 0.0, atol=1e-7)
