from __future__ import annotations

from dataclasses import replace
from math import cos, sin

import numpy as np

from flightrl.navigation.mission_spec import TargetAnchor
from flightrl.navigation.room_generation import (
    SemanticRoomGenerationConfig,
    generate_semantic_room,
)
from flightrl.navigation.semantic_scene import Bounds3D, SemanticObject, SemanticScene
from flightrl.sixdof.orientation import quat_to_yaw

from .mujoco_env import MuJoCoCoverageEnv
from .student_sequence import (
    CoverageSequenceDataset,
    EVENT_ADVANCE,
    EVENT_CONTINUE_SCAN,
    EVENT_ENTER_SCAN,
    EVENT_RESUME_ADVANCE,
    coverage_sequence_metadata,
    require_coverage_sequence_dataset,
    require_matched_counterfactual_pairs,
)
from .teacher import ScanAdvanceTeacher


def collect_teacher_dataset(
    scene_ids: tuple[int, ...],
    *,
    split: str,
    maximum_steps: int,
) -> CoverageSequenceDataset:
    if not scene_ids or any(type(seed) is not int or seed < 0 for seed in scene_ids):
        raise ValueError("coverage collection scene IDs must be non-negative integers")
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("coverage collection scene IDs must be unique")
    if type(maximum_steps) is not int or maximum_steps <= 0:
        raise ValueError("coverage collection maximum steps must be positive")
    episodes = []
    for seed in scene_ids:
        scene = generate_semantic_room(
            seed,
            SemanticRoomGenerationConfig.for_profile("diverse"),
        )
        episodes.append(_collect_episode(scene, seed=seed, maximum_steps=maximum_steps))
    dataset = CoverageSequenceDataset(
        packed_frames=np.stack([episode["packed"] for episode in episodes], axis=1),
        telemetry=np.stack([episode["telemetry"] for episode in episodes], axis=1),
        teacher_actions=np.stack([episode["actions"] for episode in episodes], axis=1),
        resets=np.stack([episode["resets"] for episode in episodes], axis=1),
        dones=np.stack([episode["dones"] for episode in episodes], axis=1),
        front_clearance_m=np.stack(
            [episode["clearance"] for episode in episodes], axis=1
        ),
        event_labels=np.stack([episode["events"] for episode in episodes], axis=1),
        scene_ids=np.asarray(scene_ids, dtype=np.uint32),
        pair_ids=np.full((maximum_steps, len(scene_ids)), -1, dtype=np.int64),
        metadata=coverage_sequence_metadata(
            split=split,
            steps=maximum_steps,
            scene_ids=scene_ids,
        ),
    )
    require_coverage_sequence_dataset(dataset)
    return dataset


def collect_matched_counterfactual_pair(
    *, seed: int, split: str
) -> CoverageSequenceDataset:
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("coverage counterfactual seed must fit uint31")
    generated = generate_semantic_room(
        seed,
        SemanticRoomGenerationConfig(obstacle_count_range=(0, 0)),
    )
    clear_scene = replace(
        generated,
        objects=tuple(obj for obj in generated.objects if not obj.collision),
    )
    clear_env = MuJoCoCoverageEnv(clear_scene, seed=seed, maximum_episode_steps=1)
    blocked_env = None
    try:
        clear_observation, _info = clear_env.reset(seed=seed)
        position = clear_env.sim.position[0].copy()
        yaw = float(quat_to_yaw(clear_env.sim.quaternion)[0])
        blocked_scene = replace(
            clear_scene,
            objects=(*clear_scene.objects, _front_blocker(clear_scene, position, yaw)),
        )
        blocked_env = MuJoCoCoverageEnv(
            blocked_scene, seed=seed, maximum_episode_steps=1
        )
        blocked_env.reset(seed=seed)
        _match_reset_state(clear_env, blocked_env)
        blocked_observation = blocked_env._observation()
        observations = np.stack((clear_observation, blocked_observation))
        telemetry = observations[:, 3072:].astype(np.float32, copy=True)
        if not np.array_equal(telemetry[0], telemetry[1]):
            raise RuntimeError(
                "coverage counterfactual reset telemetry is not identical"
            )
        clear_action, clear_event = _teacher_label(clear_env)
        blocked_action, blocked_event = _teacher_label(blocked_env)
        clearance = np.asarray(
            (clear_env.sim.ranges_m[0, 0], blocked_env.sim.ranges_m[0, 0]),
            dtype=np.float32,
        )
    finally:
        clear_env.close()
        if blocked_env is not None:
            blocked_env.close()
    scene_ids = (seed, seed)
    dataset = CoverageSequenceDataset(
        packed_frames=_pack_observations(observations)[None, ...],
        telemetry=telemetry[None, ...],
        teacher_actions=np.stack((clear_action, blocked_action))[None, ...],
        resets=np.ones((1, 2), dtype=np.uint8),
        dones=np.ones((1, 2), dtype=np.uint8),
        front_clearance_m=clearance[None, ...],
        event_labels=np.asarray(((clear_event, blocked_event),), dtype=np.uint8),
        scene_ids=np.asarray(scene_ids, dtype=np.uint32),
        pair_ids=np.zeros((1, 2), dtype=np.int64),
        metadata=coverage_sequence_metadata(
            split=split,
            steps=1,
            scene_ids=scene_ids,
        ),
    )
    require_coverage_sequence_dataset(dataset)
    require_matched_counterfactual_pairs(dataset)
    return dataset


def _collect_episode(
    scene: SemanticScene, *, seed: int, maximum_steps: int
) -> dict[str, np.ndarray]:
    env = MuJoCoCoverageEnv(scene, seed=seed, maximum_episode_steps=maximum_steps)
    try:
        observation, _info = env.reset(seed=seed)
        teacher = ScanAdvanceTeacher()
        teacher.reset(
            env.sim.position[0, :2],
            yaw_rad=float(quat_to_yaw(env.sim.quaternion)[0]),
        )
        packed = np.empty((maximum_steps, 1536), dtype=np.uint8)
        telemetry = np.empty((maximum_steps, 19), dtype=np.float32)
        actions = np.empty((maximum_steps, 2), dtype=np.float32)
        resets = np.zeros(maximum_steps, dtype=np.uint8)
        dones = np.zeros(maximum_steps, dtype=np.uint8)
        clearance = np.empty(maximum_steps, dtype=np.float32)
        events = np.empty(maximum_steps, dtype=np.uint8)
        resets[0] = 1
        for step in range(maximum_steps):
            packed[step] = _pack_observations(observation[None, :])[0]
            telemetry[step] = observation[3072:]
            previous_phase = teacher.phase
            front = float(env.sim.ranges_m[0, 0])
            action = teacher.action(
                env.sim.position[0, :2],
                yaw_rad=float(quat_to_yaw(env.sim.quaternion)[0]),
                horizontal_ranges_m=env.sim.ranges_m[0, :4],
            )
            actions[step] = action[(0, 3),]
            clearance[step] = front
            events[step] = _event(previous_phase, teacher.phase)
            observation, _reward, terminal, truncated, _info = env.step(action)
            done = terminal or truncated
            dones[step] = done
            if done and step + 1 != maximum_steps:
                raise RuntimeError(
                    "coverage teacher episode ended before collection horizon"
                )
        if not dones[-1]:
            raise RuntimeError("coverage teacher episode did not close at its horizon")
        return {
            "packed": packed,
            "telemetry": telemetry,
            "actions": actions,
            "resets": resets,
            "dones": dones,
            "clearance": clearance,
            "events": events,
        }
    finally:
        env.close()


def _teacher_label(env: MuJoCoCoverageEnv) -> tuple[np.ndarray, int]:
    teacher = ScanAdvanceTeacher()
    teacher.reset(
        env.sim.position[0, :2],
        yaw_rad=float(quat_to_yaw(env.sim.quaternion)[0]),
    )
    previous = teacher.phase
    action = teacher.action(
        env.sim.position[0, :2],
        yaw_rad=float(quat_to_yaw(env.sim.quaternion)[0]),
        horizontal_ranges_m=env.sim.ranges_m[0, :4],
    )
    return action[(0, 3),].copy(), _event(previous, teacher.phase)


def _event(previous: str, current: str) -> int:
    if previous == "advance" and current == "scan_turn":
        return EVENT_ENTER_SCAN
    if previous == "scan_turn" and current == "advance":
        return EVENT_RESUME_ADVANCE
    return EVENT_CONTINUE_SCAN if current == "scan_turn" else EVENT_ADVANCE


def _pack_observations(observations: np.ndarray) -> np.ndarray:
    frames = np.asarray(observations, dtype=np.float32)[:, :3072]
    levels = frames * 15.0
    if not np.allclose(levels, np.rint(levels), atol=1.0e-6, rtol=0.0):
        raise ValueError("coverage collection frame is not exact gray4")
    nibbles = np.rint(levels).astype(np.uint8)
    return (nibbles[:, 0::2] << 4) | nibbles[:, 1::2]


def _front_blocker(
    scene: SemanticScene, position: np.ndarray, yaw: float
) -> SemanticObject:
    center_x = float(position[0] + 0.50 * cos(yaw))
    center_y = float(position[1] + 0.50 * sin(yaw))
    z_low = max(scene.room.minimum[2] + 0.05, scene.flight_altitude_m - 0.40)
    z_high = min(scene.room.maximum[2] - 0.05, scene.flight_altitude_m + 0.40)
    return SemanticObject(
        object_id="counterfactual_blocker",
        category="obstacle",
        bounds=Bounds3D(
            (center_x - 0.14, center_y - 0.14, z_low),
            (center_x + 0.14, center_y + 0.14, z_high),
        ),
        preferred_anchor=TargetAnchor.CENTER,
        collision=True,
        rgba=(0.08, 0.08, 0.08, 1.0),
    )


def _match_reset_state(
    source: MuJoCoCoverageEnv, destination: MuJoCoCoverageEnv
) -> None:
    source_data = source.sim.data[0]
    destination_data = destination.sim.data[0]
    destination_data.qpos[:] = source_data.qpos
    destination_data.qvel[:] = source_data.qvel
    destination.sim.mujoco.mj_forward(destination.sim.model, destination_data)
    destination.sim._sync_state_from_data()
    destination.sim._update_ranges()
    destination.sim.target_position[:] = source.sim.target_position
    destination.sim.target_yaw[:] = source.sim.target_yaw
    destination.sim.command_state.fill(0.0)
    destination.sim.previous_action.fill(0.0)
    destination.sim.step_count.fill(0)
    destination.mission_origin_position[:] = source.mission_origin_position
    destination.mission_origin_yaw[:] = source.mission_origin_yaw
    destination.previous_edge_action.fill(0.0)
