from __future__ import annotations

import numpy as np

from flightrl.navigation.room_generation import (
    SemanticRoomGenerationConfig,
    generate_semantic_room,
)
from flightrl.sixdof.orientation import quat_to_yaw

from .mujoco_env import MuJoCoCoverageEnv
from .teacher import ScanAdvanceTeacher


SCHEMA = "flightrl.scan_advance_behavior_audit.v1"


def audit_scan_advance_behavior(
    seeds: tuple[int, ...],
    *,
    maximum_steps: int = 1_800,
) -> dict[str, object]:
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("scan-advance seeds must be non-empty non-negative integers")
    if type(maximum_steps) is not int or maximum_steps <= 0:
        raise ValueError("scan-advance maximum steps must be a positive integer")
    episodes = [_audit_seed(seed, maximum_steps) for seed in seeds]
    return {
        "schema": SCHEMA,
        "seeds": list(seeds),
        "maximum_steps": maximum_steps,
        "control_period_s": 0.02,
        "episodes": episodes,
        "recognizable_behavior_passed": all(
            episode["recognizable_behavior_passed"] for episode in episodes
        ),
        "evaluation_kind": "paired_privileged_teacher_simulation_sanity",
        "privileged_teacher_evaluated": True,
        "learned_policy_evaluated": False,
        "camera_causality_evaluated": False,
        "shadow_authority": False,
        "deployment_authority": False,
        "flight_authority": False,
    }


def _audit_seed(seed: int, maximum_steps: int) -> dict[str, object]:
    scene = generate_semantic_room(
        seed,
        SemanticRoomGenerationConfig.for_profile("diverse"),
    )
    env = MuJoCoCoverageEnv(
        scene,
        seed=seed,
        maximum_episode_steps=maximum_steps,
    )
    try:
        teacher = _run_episode(env, seed, teacher=ScanAdvanceTeacher())
        stationary = _run_episode(env, seed, teacher=None)
    finally:
        env.close()
    passed = (
        not teacher["safety_terminal"]
        and teacher["completed_turns"] >= 1
        and teacher["path_length_m"] >= 0.75
        and teacher["path_length_m"] >= stationary["path_length_m"] + 0.50
        and teacher["coverage_score"] >= stationary["coverage_score"] + 0.05
        and teacher["gray4_contract_valid"]
    )
    return {
        "seed": seed,
        "teacher": teacher,
        "stationary": stationary,
        "recognizable_behavior_passed": bool(passed),
    }


def _run_episode(
    env: MuJoCoCoverageEnv,
    seed: int,
    *,
    teacher: ScanAdvanceTeacher | None,
) -> dict[str, object]:
    observation, info = env.reset(seed=seed)
    yaw = float(quat_to_yaw(env.sim.quaternion)[0])
    if teacher is not None:
        teacher.reset(env.sim.position[0, :2], yaw_rad=yaw)
    positions = [env.sim.position[0].copy()]
    yaws = [yaw]
    measured_horizontal_speeds = [float(np.linalg.norm(env.sim.velocity[0, :2]))]
    commanded_forward = []
    commanded_yaw = []
    total_reward = 0.0
    minimum_clearance = float(np.min(env.sim.ranges_m[0, :4]))
    gray4_valid = _gray4_valid(observation)
    phase_changes = 0
    previous_phase = teacher.phase if teacher is not None else "stationary"
    terminal = False
    truncated = False
    for _step in range(env.maximum_episode_steps):
        if teacher is None:
            action = np.zeros(4, dtype=np.float32)
        else:
            action = teacher.action(
                env.sim.position[0, :2],
                yaw_rad=float(quat_to_yaw(env.sim.quaternion)[0]),
                horizontal_ranges_m=env.sim.ranges_m[0, :4],
            )
            if teacher.phase != previous_phase:
                phase_changes += 1
                previous_phase = teacher.phase
        observation, reward, terminal, truncated, info = env.step(action)
        total_reward += reward
        commanded_forward.append(abs(float(action[0])) * 0.25)
        commanded_yaw.append(abs(float(action[3])) * 8.0)
        positions.append(env.sim.position[0].copy())
        yaws.append(float(quat_to_yaw(env.sim.quaternion)[0]))
        measured_horizontal_speeds.append(
            float(np.linalg.norm(env.sim.velocity[0, :2]))
        )
        minimum_clearance = min(
            minimum_clearance,
            float(info["minimum_horizontal_clearance_m"]),
        )
        gray4_valid = gray4_valid and _gray4_valid(observation)
        if terminal or truncated:
            break

    path = np.asarray(positions, dtype=np.float32)
    unwrapped_yaw = np.unwrap(np.asarray(yaws, dtype=np.float32))
    stride = max(1, len(path) // 64)
    samples = path[::stride]
    if not np.array_equal(samples[-1], path[-1]):
        samples = np.vstack((samples, path[-1]))
    return {
        "controller": "privileged_scan_advance" if teacher is not None else "stationary",
        "steps": len(path) - 1,
        "safety_terminal": bool(terminal),
        "truncated": bool(truncated),
        "collision": bool(info["collision"]),
        "boundary_violation": bool(info["boundary_violation"]),
        "coverage_score": float(info["coverage_score"]),
        "visited_fraction": float(info["visited_fraction"]),
        "visible_free_fraction": float(info["visible_free_fraction"]),
        "return": float(total_reward),
        "path_length_m": float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()),
        "net_displacement_m": float(np.linalg.norm(path[-1, :2] - path[0, :2])),
        "yaw_travel_deg": float(np.rad2deg(np.abs(np.diff(unwrapped_yaw)).sum())),
        "altitude_span_m": float(np.ptp(path[:, 2])),
        "minimum_horizontal_clearance_m": minimum_clearance,
        "maximum_commanded_forward_speed_m_s": max(commanded_forward, default=0.0),
        "maximum_commanded_yaw_rate_deg_s": max(commanded_yaw, default=0.0),
        "maximum_measured_horizontal_speed_m_s": max(measured_horizontal_speeds),
        "phase_changes": phase_changes,
        "completed_turns": teacher.completed_turns if teacher is not None else 0,
        "gray4_contract_valid": gray4_valid,
        "trajectory_samples_xyz_m": samples.tolist(),
        "privileged_ranges_used_by_controller": teacher is not None,
        "actor_observation_used": False,
    }


def _gray4_valid(observation: np.ndarray) -> bool:
    levels = observation[: 64 * 48] * 15.0
    return bool(
        np.isfinite(levels).all()
        and np.all((levels >= 0.0) & (levels <= 15.0))
        and np.allclose(levels, np.rint(levels), atol=1.0e-6, rtol=0.0)
    )
