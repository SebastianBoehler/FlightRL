from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from flightrl.exploration.behavior_audit import audit_scan_advance_behavior
from flightrl.exploration.mujoco_env import MuJoCoCoverageEnv, coverage_reward
from flightrl.exploration.observation import (
    build_coverage_observation,
    coverage_action_to_edge_feedback,
)
from flightrl.exploration.teacher import ScanAdvanceTeacher
from flightrl.mujoco import is_mujoco_available, is_mujoco_rendering_available
from flightrl.navigation.room_generation import (
    SemanticRoomGenerationConfig,
    generate_semantic_room,
)


def test_coverage_observation_matches_edge_telemetry_units_without_target() -> None:
    frame = np.zeros((1, 48, 64), dtype=np.uint8)
    frame[0, 0, 0] = 17
    observation = build_coverage_observation(
        frame,
        position=np.asarray([[0.4, -0.2, 0.75]], dtype=np.float32),
        velocity=np.asarray([[0.2, -0.1, 0.05]], dtype=np.float32),
        quaternion=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        body_rates=np.asarray([[1.0, -2.0, 2.0]], dtype=np.float32),
        takeoff_origin_z=0.1,
        mission_origin_position=np.asarray([[0.1, 0.2, 0.5]], dtype=np.float32),
        mission_origin_yaw=np.asarray([0.0], dtype=np.float32),
        previous_edge_action=np.asarray([[0.25, -0.5, 0.75, -1.0]], dtype=np.float32),
    )

    assert observation.shape == (1, 3091)
    assert observation.dtype == np.float32
    assert observation[0, 0] == pytest.approx(1.0 / 15.0)
    np.testing.assert_allclose(
        observation[0, 3072:],
        (
            0.2, -0.1, 0.1,
            1.0 / 6.0, -2.0 / 6.0, 0.5,
            0.0, 0.0, 1.0,
            0.65 / 2.5,
            0.3 / 4.0, -0.4 / 4.0, 0.25 / 2.0,
            0.0, 1.0,
            0.25, -0.5, 0.75, -1.0,
        ),
        atol=1.0e-6,
    )


def test_coverage_yaw_action_feedback_retains_edge_45_degree_unit() -> None:
    feedback = coverage_action_to_edge_feedback(
        np.asarray([0.4, 0.0, 0.0, 0.5], dtype=np.float32)
    )

    np.testing.assert_allclose(feedback, (0.4, 0.0, 0.0, 4.0 / 45.0))


def test_mujoco_coverage_env_has_visible_motion_contract_without_auto_reset() -> None:
    if not is_mujoco_available() or not is_mujoco_rendering_available():
        pytest.skip("MuJoCo rendering is unavailable")
    scene = generate_semantic_room(
        512,
        SemanticRoomGenerationConfig(obstacle_count_range=(2, 2)),
    )
    env = MuJoCoCoverageEnv(scene, seed=512, maximum_episode_steps=1)
    try:
        observation, reset_info = env.reset(seed=512)
        assert observation.shape == (3091,)
        assert np.all((observation[:3072] * 15) % 1 == 0)
        assert env.sim.position[0, 2] == pytest.approx(scene.flight_altitude_m)
        assert reset_info["actor_observation_contains_range"] is False
        assert reset_info["actor_observation_contains_map"] is False
        np.testing.assert_array_equal(env.action_space.low, (-1.0, 0.0, 0.0, -1.0))
        np.testing.assert_array_equal(env.action_space.high, (1.0, 0.0, 0.0, 1.0))

        next_observation, reward, terminal, truncated, info = env.step(
            np.asarray([0.5, 0.0, 0.0, 1.0], dtype=np.float32)
        )

        assert np.isfinite(reward)
        assert terminal is False
        assert truncated is True
        assert next_observation[3072 + 15] == pytest.approx(0.5)
        assert next_observation[3072 + 18] == pytest.approx(8.0 / 45.0)
        assert info["maximum_yaw_rate_deg_s"] == 8.0
        assert info["collision"] is False
        assert info["boundary_violation"] is False
        assert info["flight_authority"] is False
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(4, dtype=np.float32))
        with pytest.raises(ValueError, match="structurally zero"):
            env.reset(seed=512)
            env.step(np.asarray([0.0, 0.1, 0.0, 0.0], dtype=np.float32))
    finally:
        env.close()


def test_coverage_reward_is_normalized_and_collision_dominates_all_discovery() -> None:
    assert coverage_reward(0.0, safety_terminal=False) == 0.0
    assert coverage_reward(1.0, safety_terminal=False) == 1.0
    assert coverage_reward(1.0, safety_terminal=True) < 0.0


def test_mujoco_coverage_env_passes_gymnasium_checker() -> None:
    if not is_mujoco_available() or not is_mujoco_rendering_available():
        pytest.skip("MuJoCo rendering is unavailable")
    scene = generate_semantic_room(
        513,
        SemanticRoomGenerationConfig(obstacle_count_range=(2, 2)),
    )
    env = MuJoCoCoverageEnv(scene, seed=513, maximum_episode_steps=2)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_scan_advance_audit_pairs_teacher_with_same_seed_stationary_baseline() -> None:
    if not is_mujoco_available() or not is_mujoco_rendering_available():
        pytest.skip("MuJoCo rendering is unavailable")

    report = audit_scan_advance_behavior((514,), maximum_steps=2)

    assert report["seeds"] == [514]
    assert report["episodes"][0]["teacher"]["controller"] == "privileged_scan_advance"
    assert report["episodes"][0]["stationary"]["controller"] == "stationary"
    assert report["episodes"][0]["teacher"]["gray4_contract_valid"] is True
    assert report["learned_policy_evaluated"] is False
    assert report["flight_authority"] is False


def test_scan_advance_teacher_produces_recognizable_forward_leg_and_turn() -> None:
    teacher = ScanAdvanceTeacher()
    teacher.reset(np.asarray([0.0, 0.0], dtype=np.float32), yaw_rad=0.0)

    forward = teacher.action(
        np.asarray([0.0, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        horizontal_ranges_m=np.asarray([2.0, 2.0, 1.0, 0.4]),
    )
    start_turn = teacher.action(
        np.asarray([0.81, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        horizontal_ranges_m=np.asarray([2.0, 2.0, 1.0, 0.4]),
    )
    continue_turn = teacher.action(
        np.asarray([0.81, 0.0], dtype=np.float32),
        yaw_rad=np.pi / 4.0,
        horizontal_ranges_m=np.asarray([2.0, 2.0, 1.0, 0.4]),
    )
    blocked_after_ninety = teacher.action(
        np.asarray([0.81, 0.0], dtype=np.float32),
        yaw_rad=np.pi / 2.0,
        horizontal_ranges_m=np.asarray([0.4, 2.0, 1.0, 0.4]),
    )
    next_leg = teacher.action(
        np.asarray([0.81, 0.0], dtype=np.float32),
        yaw_rad=np.pi / 2.0 + 0.1,
        horizontal_ranges_m=np.asarray([1.0, 2.0, 1.0, 0.4]),
    )

    np.testing.assert_array_equal(forward, (0.5, 0.0, 0.0, 0.0))
    np.testing.assert_array_equal(start_turn, (0.0, 0.0, 0.0, 1.0))
    np.testing.assert_array_equal(continue_turn, start_turn)
    np.testing.assert_array_equal(blocked_after_ninety, start_turn)
    np.testing.assert_array_equal(next_leg, forward)
    assert teacher.completed_turns == 1
    assert teacher.phase == "advance"


def test_scan_advance_teacher_ignores_unobservable_side_ranges_and_turns_positive() -> None:
    teacher = ScanAdvanceTeacher()
    teacher.reset(np.zeros(2, dtype=np.float32), yaw_rad=0.0)

    first_turn = teacher.action(
        np.zeros(2, dtype=np.float32),
        yaw_rad=0.0,
        horizontal_ranges_m=np.asarray([0.4, 2.0, 0.5, 1.5]),
    )
    teacher.action(
        np.zeros(2, dtype=np.float32),
        yaw_rad=np.pi / 2.0,
        horizontal_ranges_m=np.asarray([2.0, 2.0, 0.5, 1.5]),
    )
    second_turn = teacher.action(
        np.zeros(2, dtype=np.float32),
        yaw_rad=np.pi / 2.0,
        horizontal_ranges_m=np.asarray([0.4, 2.0, 1.5, 0.5]),
    )

    np.testing.assert_array_equal(first_turn, (0.0, 0.0, 0.0, 1.0))
    np.testing.assert_array_equal(second_turn, (0.0, 0.0, 0.0, 1.0))
    assert teacher.phase == "scan_turn"
    assert teacher.privileged_inputs == ("front_range_m",)
    assert teacher.privileged_teacher is True
    assert teacher.flight_authority is False
