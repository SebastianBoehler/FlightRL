from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flightrl.hardware.aideck_stream import AiDeckFrame
from flightrl.mujoco import is_mujoco_available
from flightrl.mujoco.semantic_exploration import line_of_sight_clear
from flightrl.mujoco.semantic_observation import (
    GROUNDING_CONFIDENCE_INDEX,
    GROUNDING_HORIZONTAL_ERROR_INDEX,
)
from flightrl.mujoco.semantic_planning import PrivilegedGridPlanner
from flightrl.navigation.mission_spec import TargetAnchor
from flightrl.navigation.room_generation import (
    SemanticRoomGenerationConfig,
    generate_semantic_room,
)
from flightrl.navigation.semantic_scene import (
    Bounds3D,
    SemanticObject,
    SemanticScene,
)
from flightrl.semantic.contract import (
    GroundingDetection,
    GroundingResult,
    NormalizedBox,
)
from flightrl.semantic.fast_policy import FastSemanticPolicyClock
from flightrl.vision import VisionObservationConfig


def test_active_room_generation_adds_seeded_interior_obstacles() -> None:
    config = SemanticRoomGenerationConfig(obstacle_count_range=(3, 3))

    scene = generate_semantic_room(81, config)
    repeated = generate_semantic_room(81, config)
    obstacles = [obj for obj in scene.objects if obj.category == "obstacle"]

    assert scene == repeated
    assert len(obstacles) == 3
    assert all(obj.collision for obj in obstacles)
    for semantic_object in scene.objects:
        if semantic_object.approach_position_m is None:
            continue
        assert all(
            not (
                obstacle.bounds.minimum[0] - 0.4
                <= semantic_object.approach_position_m[0]
                <= obstacle.bounds.maximum[0] + 0.4
                and obstacle.bounds.minimum[1] - 0.4
                <= semantic_object.approach_position_m[1]
                <= obstacle.bounds.maximum[1] + 0.4
            )
            for obstacle in obstacles
        )


def test_privileged_planner_routes_around_inflated_obstacle() -> None:
    scene = generate_semantic_room(
        84,
        SemanticRoomGenerationConfig(obstacle_count_range=(3, 3)),
    )
    planner = PrivilegedGridPlanner(scene)

    path = planner.path(
        np.asarray((-1.5, -1.5), dtype=np.float32),
        np.asarray((1.5, 1.5), dtype=np.float32),
    )

    assert len(path) > 1
    assert all(not planner.blocked[planner.nearest_free_cell(point)] for point in path)


def test_privileged_coverage_goals_are_free_interior_viewpoints() -> None:
    scene = generate_semantic_room(
        86,
        SemanticRoomGenerationConfig.for_profile("diverse"),
    )
    planner = PrivilegedGridPlanner(scene)

    goals = planner.coverage_goals()

    assert len(goals) == 5
    assert all(not planner.blocked[planner.nearest_free_cell(goal)] for goal in goals)
    assert all(scene.room.minimum[0] + 0.5 < goal[0] for goal in goals)
    assert all(goal[0] < scene.room.maximum[0] - 0.5 for goal in goals)
    assert all(scene.room.minimum[1] + 0.5 < goal[1] for goal in goals)
    assert all(goal[1] < scene.room.maximum[1] - 0.5 for goal in goals)


def test_collision_geometry_occludes_semantic_target() -> None:
    room = Bounds3D((-2.0, -2.0, 0.0), (3.0, 2.0, 2.5))
    target = SemanticObject(
        "monitor_0",
        "monitor",
        Bounds3D((2.0, -0.3, 0.7), (2.1, 0.3, 1.4)),
        preferred_anchor=TargetAnchor.APPROACH,
        approach_position_m=(1.5, 0.0, 1.0),
        collision=False,
    )
    blocker = SemanticObject(
        "obstacle_0",
        "obstacle",
        Bounds3D((0.8, -0.4, 0.0), (1.2, 0.4, 1.8)),
        preferred_anchor=TargetAnchor.CENTER,
        collision=True,
    )
    scene = SemanticScene(room, (target, blocker), flight_altitude_m=1.0)

    assert not line_of_sight_clear(
        scene,
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray(target.bounds.center),
        ignored_object_id=target.object_id,
    )


def test_active_policy_allows_forward_only_before_detection() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.semantic_training import (
        SemanticTrainingEnv,
        SimulatedSemanticDetectorConfig,
    )
    from flightrl.mujoco.semantic_vision_policy import SemanticVisionPolicy

    env = SemanticTrainingEnv(
        room_seed=83,
        num_envs=1,
        seed=83,
        active_exploration=True,
        detector=SimulatedSemanticDetectorConfig(dropout_probability=1.0),
    )
    try:
        observations, _ = env.reset(83)
        policy = SemanticVisionPolicy(env, hidden_size=32)
        with torch.no_grad():
            policy.decoder.decoder_mean.weight.zero_()
            policy.decoder.decoder_mean.bias.fill_(0.5)
            assert policy.recurrent_safety is not None
            policy.recurrent_safety.clearance_head.weight.zero_()
            policy.recurrent_safety.clearance_head.bias.fill_(2.0)
            policy.recurrent_safety.collision_risk_head.weight.zero_()
            policy.recurrent_safety.collision_risk_head.bias.fill_(-5.0)
        distribution, _ = policy(torch.from_numpy(observations))
        visible_observations = observations.copy()
        visible_state = visible_observations[:, env.layout.proprioception_slice]
        visible_state[:, GROUNDING_CONFIDENCE_INDEX] = 0.9
        visible_state[:, GROUNDING_HORIZONTAL_ERROR_INDEX] = 0.0
        visible_distribution, _ = policy(torch.from_numpy(visible_observations))
        visible_state[:, GROUNDING_HORIZONTAL_ERROR_INDEX] = 1.0
        off_center_distribution, _ = policy(torch.from_numpy(visible_observations))
        env.backend.record_target_observation(
            0,
            bearing_rad=0.0,
            distance_m=1.0,
            confidence=0.9,
        )
        env.backend._write_observations()
        env._write_grounding_observations()
        memory_distribution, _ = policy(torch.from_numpy(env.observations.copy()))
        with torch.no_grad():
            policy.recurrent_safety.collision_risk_head.bias.fill_(2.0)
        risky_distribution, _ = policy(torch.from_numpy(observations))
    finally:
        env.close()

    expected_forward = (
        torch.sigmoid(torch.tensor(0.5))
        * torch.sigmoid(10.0 * (4.0 * torch.sigmoid(torch.tensor(2.0)) - 0.65))
        * torch.sigmoid(16.0 * (0.35 - torch.sigmoid(torch.tensor(-5.0))))
    )
    assert float(distribution.mean[0, 0].detach()) == pytest.approx(
        float(expected_forward)
    )
    assert float(visible_distribution.mean[0, 0].detach()) == pytest.approx(
        float(distribution.mean[0, 0].detach())
    )
    assert float(off_center_distribution.mean[0, 0].detach()) == 0.0
    assert float(memory_distribution.mean[0, 0].detach()) == pytest.approx(
        float(distribution.mean[0, 0].detach())
    )
    assert abs(float(memory_distribution.mean[0, 3].detach())) <= 20.0 / 60.0
    assert float(risky_distribution.mean[0, 0].detach()) < 0.001
    assert torch.equal(distribution.mean[0, 1:3], torch.zeros(2))
    assert float(distribution.mean[0, 3].detach()) == pytest.approx(20.0 / 60.0)


def test_active_environment_accepts_higher_resolution_vision_contract() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.semantic_training import SemanticTrainingEnv

    env = SemanticTrainingEnv(
        room_seed=85,
        num_envs=1,
        seed=85,
        active_exploration=True,
        vision_config=VisionObservationConfig(
            width=128,
            height=96,
            color_mode="grayscale",
            frame_stack=1,
            include_delta=True,
            include_motion_mask=True,
            normalization="minus_one_one",
        ),
    )
    try:
        observations, _ = env.reset(85)
    finally:
        env.close()

    assert env.vision_config.shape == (3, 96, 128)
    assert observations.shape == (1, env.layout.flat_dim)


def test_active_expert_explores_without_oracle_observation() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo optional dependency is not installed")
    from flightrl.mujoco.semantic_training import (
        SemanticTrainingEnv,
        SimulatedSemanticDetectorConfig,
    )

    env = SemanticTrainingEnv(
        room_seed=89,
        num_envs=1,
        seed=89,
        active_exploration=True,
        detector=SimulatedSemanticDetectorConfig(dropout_probability=1.0),
    )
    try:
        env.reset(89)
        actions = env.expert_actions()
    finally:
        env.close()

    assert not env.target_observed[0]
    assert np.linalg.norm(actions[0, (0, 3)]) > 0.0
    assert np.array_equal(actions[0, 1:3], np.zeros(2, dtype=np.float32))
    assert abs(actions[0, 3]) <= 20.0 / env.control.max_yawrate_deg_s


def test_fast_policy_clock_advances_raw_frames_without_reusing_map_update() -> None:
    policy = _RecordingPolicy()
    frame = _frame(1, 10.0)
    grounding = _grounding(1, 10.0)
    pipeline = SimpleNamespace(
        latest_frame=lambda: frame,
        latest=lambda: (frame, grounding),
    )
    clock = FastSemanticPolicyClock(policy, "monitor")

    first = clock.poll(pipeline, {})
    frame = _frame(2, 10.05)
    second = clock.poll(pipeline, {})

    assert first is not None and second is not None
    assert clock.raw_frames_processed == 2
    assert clock.grounding_updates == 1
    assert policy.map_updates == [True, False]


class _RecordingPolicy:
    def __init__(self) -> None:
        self.map_updates: list[bool] = []

    def step(self, **kwargs) -> dict:
        self.map_updates.append(bool(kwargs["update_semantic_memory"]))
        return {"controls_drone": False}


def _frame(index: int, host_time_s: float) -> AiDeckFrame:
    return AiDeckFrame(
        index,
        host_time_s,
        64,
        48,
        1,
        2,
        np.zeros((48, 64), dtype=np.uint8),
    )


def _grounding(index: int, host_time_s: float) -> GroundingResult:
    return GroundingResult(
        prompt="monitor",
        frame_index=index,
        frame_host_time_s=host_time_s,
        image_width=64,
        image_height=48,
        source_mean=50.0,
        inference_ms=10.0,
        detections=(
            GroundingDetection(
                "monitor",
                0.8,
                NormalizedBox(0.3, 0.2, 0.7, 0.8),
            ),
        ),
    )
