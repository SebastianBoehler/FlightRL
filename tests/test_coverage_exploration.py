from __future__ import annotations

import numpy as np
import pytest
import torch

from flightrl.exploration.contract import coverage_contract_payload
from flightrl.exploration.coverage import CoverageTracker
from flightrl.exploration.objective_audit import audit_coverage_objective
from flightrl.exploration.policy import CoverageExplorationActor
from flightrl.navigation.room_generation import (
    SemanticRoomGenerationConfig,
    generate_semantic_room,
)


def test_coverage_actor_contract_excludes_privileged_mapping_and_ranges() -> None:
    contract = coverage_contract_payload()

    assert contract["contract_id"] == "aideck-coverage-policy-v1"
    assert contract["observation"]["flat_values"] == 3091
    assert contract["observation"]["segments"] == {
        "current_gray4": [0, 3072],
        "telemetry": [3072, 3091],
    }
    telemetry = contract["observation"]["telemetry"]["order"]
    assert not any("range" in name for name in telemetry)
    assert contract["observation"]["prohibited_actor_inputs"] == [
        "target_token",
        "target_pose",
        "object_detection",
        "range_rays",
        "occupancy_grid",
        "scene_geometry",
        "privileged_pose",
    ]
    assert contract["action"]["controlled_axes"] == ["vx", "yaw_rate"]
    assert contract["action"]["structurally_zero_axes"] == ["vy", "vz"]
    assert contract["action"]["maximum_yaw_rate_deg_s"] == 8.0
    assert contract["authority"] == "simulation_only"


def test_coverage_tracker_rewards_cells_once_and_resets() -> None:
    scene = generate_semantic_room(
        302,
        SemanticRoomGenerationConfig(obstacle_count_range=(2, 2)),
    )
    tracker = CoverageTracker(scene)
    start = tracker.planner.cell_center(tracker.planner.nearest_free_cell((0.0, 0.0)))

    first = tracker.update(start, yaw_rad=0.0)
    repeated = tracker.update(start, yaw_rad=0.0)

    assert first.new_visited_cells == 1
    assert first.new_visible_free_cells > 0
    assert repeated.new_visited_cells == 0
    assert repeated.new_visible_free_cells == 0
    assert not np.any(tracker.visible & tracker.planner.blocked)

    tracker.reset()

    assert tracker.visited_count == 0
    assert tracker.visible_count == 0


def test_coverage_tracker_does_not_credit_inflated_obstacle_cells() -> None:
    scene = generate_semantic_room(
        304,
        SemanticRoomGenerationConfig(obstacle_count_range=(3, 3)),
    )
    tracker = CoverageTracker(scene)
    blocked = np.argwhere(tracker.planner.blocked)
    assert len(blocked) > 0
    cell = tuple(int(value) for value in blocked[0])
    position = tracker.planner.cell_center(cell)

    step = tracker.update(position, yaw_rad=0.0)

    assert step.position_in_free_cell is False
    assert step.new_visited_cells == 0
    assert not np.any(tracker.visited & tracker.planner.blocked)

    with pytest.raises(ValueError, match="outside the scene room"):
        tracker.update(np.asarray([999.0, 999.0]), yaw_rad=0.0)


def test_coverage_tracker_exposes_dense_privileged_score_without_actor_map() -> None:
    scene = generate_semantic_room(
        303,
        SemanticRoomGenerationConfig(obstacle_count_range=(3, 3)),
    )
    tracker = CoverageTracker(scene)
    goals = tracker.planner.coverage_goals()
    previous_fraction = 0.0

    for index, goal in enumerate(goals):
        next_goal = goals[(index + 1) % len(goals)]
        yaw = float(np.arctan2(next_goal[1] - goal[1], next_goal[0] - goal[0]))
        step = tracker.update(goal, yaw_rad=yaw)
        assert step.visible_free_fraction >= previous_fraction
        previous_fraction = step.visible_free_fraction

    report = tracker.report()
    assert report["visited_cells"] == len(goals)
    assert 0.0 < report["visible_free_fraction"] <= 1.0
    assert report["actor_observation_contains_map"] is False
    assert report["flight_authority"] is False


def test_coverage_objective_distinguishes_movement_from_stationary_scan() -> None:
    report = audit_coverage_objective(tuple(range(410, 426)))

    assert report["objective_sanity_passed"] is True
    assert len(report["episodes"]) == 16
    assert report["stationary_visible_saturation_episodes"] >= 1
    assert all(
        episode["privileged_route"]["coverage_score"]
        > episode["stationary_scan"]["coverage_score"]
        for episode in report["episodes"]
    )
    assert report["learned_policy_evaluated"] is False
    assert report["flight_authority"] is False


def test_coverage_objective_never_teleports_across_disconnected_grid_regions() -> None:
    report = audit_coverage_objective((0,))

    assert report["episodes"][0]["privileged_route"]["disconnected_goals_skipped"] >= 1


def test_coverage_actor_has_no_target_input_and_preserves_applied_setpoint() -> None:
    actor = CoverageExplorationActor(hidden_size=48)
    observation = torch.zeros((2, 3091), dtype=torch.float32)
    observation[:, 3072 + 8] = 1.0
    observation[:, 3072 + 14] = 1.0
    observation[:, 3072 + 15] = 0.4
    observation[:, 3072 + 18] = 4.0 / 45.0
    state = actor.initial_state(2)

    action, next_state = actor.forward_step(observation, state)

    assert action[:, 0].tolist() == pytest.approx([0.4, 0.4])
    assert action[:, 1:3].count_nonzero().item() == 0
    assert action[:, 3].tolist() == pytest.approx([0.5, 0.5])
    assert next_state.shape == (2, 48)
    assert actor.parameter_count <= 50_000

    with pytest.raises(ValueError, match="shape"):
        actor.forward_step(torch.zeros((2, 3094)), state)
