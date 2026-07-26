from __future__ import annotations

import numpy as np

from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.action_targets import shape_action_targets
from flightrl.sixdof.env import euler_to_quat


def test_precontact_drift_brake_shapes_open_space_pitch_against_velocity() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=41, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 1.2
    env.velocity[:] = np.asarray([[1.5, 0.0, 0.0]], dtype=np.float32)
    env.quaternion[:] = euler_to_quat(np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32))
    target = np.zeros((1, 4), dtype=np.float32)

    shaped = shape_action_targets(env, target, "precontact_drift_brake")

    assert shaped[0, 2] < -0.50
    assert abs(float(shaped[0, 1])) < 1e-6


def test_precontact_drift_brake_uses_body_frame_velocity() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=42, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 1.2
    env.velocity[:] = np.asarray([[1.5, 0.0, 0.0]], dtype=np.float32)
    env.quaternion[:] = euler_to_quat(np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), np.asarray([np.pi / 2], dtype=np.float32))
    target = np.zeros((1, 4), dtype=np.float32)

    shaped = shape_action_targets(env, target, "precontact_drift_brake")

    assert shaped[0, 1] < -0.50
    assert abs(float(shaped[0, 2])) < 1e-5


def test_precontact_drift_brake_does_not_override_close_obstacle_targets() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=43, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 0.35
    env.velocity[:] = np.asarray([[1.5, 0.0, 0.0]], dtype=np.float32)
    target = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    shaped = shape_action_targets(env, target, "precontact_drift_brake")

    np.testing.assert_allclose(shaped, target)


def test_precontact_drift_brake_strength_scales_target_blend() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=44, task="obstacle_avoidance", reset_profile="obstacle_hover_live")
    env.ranges_m[:, :4] = 1.2
    env.velocity[:] = np.asarray([[1.5, 0.0, 0.0]], dtype=np.float32)
    target = np.zeros((1, 4), dtype=np.float32)

    full = shape_action_targets(env, target, "precontact_drift_brake")
    mild = shape_action_targets(env, target, "precontact_drift_brake", strength=0.25)

    assert mild[0, 2] < 0.0
    assert abs(float(mild[0, 2])) < abs(float(full[0, 2]))
