from __future__ import annotations

import numpy as np

from flightrl.navigation.room_generation import (
    SemanticRoomGenerationConfig,
    generate_semantic_room,
)

from .coverage import CoverageTracker


SCHEMA = "flightrl.coverage_objective_offline_audit.v1"
SCAN_HEADINGS = tuple(np.linspace(-np.pi, np.pi, 8, endpoint=False))


def audit_coverage_objective(seeds: tuple[int, ...]) -> dict[str, object]:
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("coverage audit seeds must be non-empty non-negative integers")
    episodes = [_audit_episode(seed) for seed in seeds]
    passed = all(
        episode["privileged_route"]["coverage_score"]
        > episode["stationary_scan"]["coverage_score"]
        and episode["privileged_route"]["visited_cells"]
        > episode["stationary_scan"]["visited_cells"]
        for episode in episodes
    )
    return {
        "schema": SCHEMA,
        "seeds": list(seeds),
        "episodes": episodes,
        "stationary_visible_saturation_episodes": sum(
            episode["stationary_scan"]["visible_free_fraction"] >= 0.999
            for episode in episodes
        ),
        "objective_sanity_passed": passed,
        "evaluation_kind": "privileged_geometry_objective_sanity_only",
        "learned_policy_evaluated": False,
        "dynamics_evaluated": False,
        "camera_causality_evaluated": False,
        "training_authority": False,
        "deployment_authority": False,
        "flight_authority": False,
    }


def _audit_episode(seed: int) -> dict[str, object]:
    scene = generate_semantic_room(
        seed,
        SemanticRoomGenerationConfig.for_profile("diverse"),
    )
    stationary = CoverageTracker(scene)
    start_cell = stationary.planner.nearest_free_cell((0.0, 0.0))
    start = stationary.planner.cell_center(start_cell)
    for yaw in SCAN_HEADINGS:
        stationary.update(start, yaw_rad=float(yaw))

    route = CoverageTracker(scene)
    current = start
    path_cells = 0
    disconnected_goals_skipped = 0
    route.update(current, yaw_rad=0.0)
    for goal in route.planner.coverage_goals():
        path = route.planner.path(current, goal)
        if not path:
            disconnected_goals_skipped += 1
            continue
        path_cells += len(path)
        for index, point in enumerate(path):
            next_point = path[min(index + 1, len(path) - 1)]
            delta = next_point - point
            yaw = float(np.arctan2(delta[1], delta[0])) if np.any(delta) else 0.0
            route.update(point, yaw_rad=yaw)
        for yaw in SCAN_HEADINGS:
            route.update(goal, yaw_rad=float(yaw))
        current = goal

    return {
        "seed": seed,
        "stationary_scan": stationary.report(),
        "privileged_route": {
            **route.report(),
            "path_cells": path_cells,
            "disconnected_goals_skipped": disconnected_goals_skipped,
            "actor_observation_used": False,
        },
    }
