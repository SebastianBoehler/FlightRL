from __future__ import annotations

import numpy as np
import torch

from flightrl.navigation.room_generation import (
    SemanticRoomGenerationConfig,
    generate_semantic_room,
)
from flightrl.puffer4_edge_schema import EDGE_FRAME_PIXELS

from .mujoco_env import MuJoCoCoverageEnv
from .policy import CoverageExplorationActor
from .teacher import ScanAdvanceTeacher


CLOSED_LOOP_SCHEMA = "flightrl.coverage.student_closed_loop.v1"
CLOSED_LOOP_COVERAGE_MARGIN = 0.05
CLOSED_LOOP_MINIMUM_PATH_M = 0.50
CLOSED_LOOP_CHALLENGE_CLEARANCE_M = ScanAdvanceTeacher.resume_clearance_m
CLOSED_LOOP_REQUIRED_CHALLENGE_RATE = 1.0
_CAMERA_HISTORY = {
    "clean": "own_current",
    "frozen": "own_first_frame_repeated",
    "history_permuted": "fixed_cyclic_scene_donor_current_frames",
}


@torch.no_grad()
def evaluate_coverage_student_closed_loop(
    actor: CoverageExplorationActor,
    *,
    scene_ids: tuple[int, ...],
    maximum_steps: int,
) -> dict[str, object]:
    if type(actor) is not CoverageExplorationActor:
        raise TypeError("coverage closed-loop actor type is incompatible")
    if (
        len(scene_ids) < 2
        or any(type(seed) is not int or seed < 0 for seed in scene_ids)
        or len(set(scene_ids)) != len(scene_ids)
    ):
        raise ValueError("coverage closed loop requires unique non-negative scene IDs")
    if type(maximum_steps) is not int or maximum_steps <= 0:
        raise ValueError("coverage closed-loop steps must be positive")
    actor.eval()
    modes = {
        mode: _run_mode(actor, scene_ids, maximum_steps, mode)
        for mode in _CAMERA_HISTORY
    }
    clean = modes["clean"]
    checks = {
        "recognizable_clean_movement": clean["mean_path_length_m"]
        >= CLOSED_LOOP_MINIMUM_PATH_M,
        "clean_coverage_above_frozen": clean["mean_coverage_score"]
        >= modes["frozen"]["mean_coverage_score"] + CLOSED_LOOP_COVERAGE_MARGIN,
        "clean_coverage_above_history_permuted": clean["mean_coverage_score"]
        >= modes["history_permuted"]["mean_coverage_score"]
        + CLOSED_LOOP_COVERAGE_MARGIN,
        "clean_has_no_safety_terminal": clean["safety_terminal_rate"] == 0.0,
        "clean_has_no_collision_or_boundary": clean["collision_rate"] == 0.0
        and clean["boundary_violation_rate"] == 0.0,
        "clean_encounters_obstacle_challenge": clean["obstacle_challenge_rate"]
        >= CLOSED_LOOP_REQUIRED_CHALLENGE_RATE,
    }
    return {
        "schema": CLOSED_LOOP_SCHEMA,
        "held_out_scene_ids": list(scene_ids),
        "maximum_steps": maximum_steps,
        "modes": modes,
        "closed_loop_gate": {
            "thresholds": {
                "minimum_clean_coverage_margin": CLOSED_LOOP_COVERAGE_MARGIN,
                "minimum_clean_path_length_m": CLOSED_LOOP_MINIMUM_PATH_M,
                "maximum_obstacle_challenge_clearance_m": (
                    CLOSED_LOOP_CHALLENGE_CLEARANCE_M
                ),
                "required_obstacle_challenge_rate": (
                    CLOSED_LOOP_REQUIRED_CHALLENGE_RATE
                ),
            },
            "checks": checks,
            "passed": all(checks.values()),
        },
        "evaluation_kind": "held_out_mujoco_closed_loop_camera_causality",
        "permutation_contract": (
            "one fixed cyclic scene donor for the complete visual history; "
            "recipient telemetry remains unmodified"
        ),
        "generalization_authority": False,
        "training_authority": False,
        "deployment_authority": False,
        "shadow_authority": False,
        "flight_authority": False,
    }


def _run_mode(
    actor: CoverageExplorationActor,
    scene_ids: tuple[int, ...],
    maximum_steps: int,
    mode: str,
) -> dict[str, object]:
    environments = []
    try:
        for seed in scene_ids:
            scene = generate_semantic_room(
                seed,
                SemanticRoomGenerationConfig.for_profile("diverse"),
            )
            environments.append(
                MuJoCoCoverageEnv(
                    scene,
                    seed=seed,
                    maximum_episode_steps=maximum_steps,
                )
            )
        observations = []
        infos = []
        positions = []
        for seed, env in zip(scene_ids, environments, strict=True):
            observation, info = env.reset(seed=seed)
            observations.append(observation)
            infos.append(info)
            positions.append(env.sim.position[0, :2].copy())
        first_frames = np.stack(observations)[:, :EDGE_FRAME_PIXELS].copy()
        active = np.ones(len(environments), dtype=bool)
        safety_terminal = np.zeros(len(environments), dtype=bool)
        steps = np.zeros(len(environments), dtype=np.int64)
        path_length = np.zeros(len(environments), dtype=np.float64)
        returns = np.zeros(len(environments), dtype=np.float64)
        minimum_clearance = np.asarray(
            [float(np.min(env.sim.ranges_m[0, :4])) for env in environments]
        )
        minimum_front_clearance = np.asarray(
            [float(env.sim.ranges_m[0, 0]) for env in environments]
        )
        state = actor.initial_state(len(environments))
        for _step in range(maximum_steps):
            model_observation = torch.from_numpy(np.stack(observations))
            if mode == "frozen":
                model_observation[:, :EDGE_FRAME_PIXELS] = torch.from_numpy(
                    first_frames
                )
            elif mode == "history_permuted":
                model_observation[:, :EDGE_FRAME_PIXELS] = model_observation[
                    :, :EDGE_FRAME_PIXELS
                ].roll(1, dims=0)
            action, state = actor.forward_step(model_observation, state)
            commands = action.cpu().numpy()
            for index, env in enumerate(environments):
                if not active[index]:
                    continue
                observation, reward, terminal, truncated, info = env.step(
                    commands[index]
                )
                next_position = env.sim.position[0, :2].copy()
                path_length[index] += float(
                    np.linalg.norm(next_position - positions[index])
                )
                positions[index] = next_position
                observations[index] = observation
                infos[index] = info
                returns[index] += reward
                steps[index] += 1
                minimum_clearance[index] = min(
                    minimum_clearance[index],
                    float(info["minimum_horizontal_clearance_m"]),
                )
                minimum_front_clearance[index] = min(
                    minimum_front_clearance[index],
                    float(env.sim.ranges_m[0, 0]),
                )
                if terminal or truncated:
                    active[index] = False
                    safety_terminal[index] = terminal
            if not np.any(active):
                break
        episodes = [
            {
                "scene_id": seed,
                "steps": int(steps[index]),
                "coverage_score": float(infos[index]["coverage_score"]),
                "path_length_m": float(path_length[index]),
                "return": float(returns[index]),
                "minimum_horizontal_clearance_m": float(minimum_clearance[index]),
                "minimum_front_clearance_m": float(
                    minimum_front_clearance[index]
                ),
                "safety_terminal": bool(safety_terminal[index]),
                "collision": bool(infos[index]["collision"]),
                "boundary_violation": bool(infos[index]["boundary_violation"]),
            }
            for index, seed in enumerate(scene_ids)
        ]
        return {
            "camera_history": _CAMERA_HISTORY[mode],
            "episodes": len(episodes),
            "mean_coverage_score": float(
                np.mean([episode["coverage_score"] for episode in episodes])
            ),
            "mean_path_length_m": float(
                np.mean([episode["path_length_m"] for episode in episodes])
            ),
            "safety_terminal_rate": float(np.mean(safety_terminal)),
            "collision_rate": float(
                np.mean([episode["collision"] for episode in episodes])
            ),
            "boundary_violation_rate": float(
                np.mean([episode["boundary_violation"] for episode in episodes])
            ),
            "obstacle_challenge_rate": float(
                np.mean(
                    [
                        episode["minimum_front_clearance_m"]
                        <= CLOSED_LOOP_CHALLENGE_CLEARANCE_M
                        for episode in episodes
                    ]
                )
            ),
            "episode_results": episodes,
        }
    finally:
        for env in environments:
            env.close()
