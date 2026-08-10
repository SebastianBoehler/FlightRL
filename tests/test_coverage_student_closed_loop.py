from __future__ import annotations

import pytest

from flightrl.exploration.policy import CoverageExplorationActor
from flightrl.exploration.student_closed_loop import (
    evaluate_coverage_student_closed_loop,
)
from flightrl.mujoco import is_mujoco_available, is_mujoco_rendering_available


pytestmark = pytest.mark.skipif(
    not is_mujoco_available() or not is_mujoco_rendering_available(),
    reason="MuJoCo rendering is unavailable",
)


def test_closed_loop_report_compares_fixed_complete_camera_histories() -> None:
    report = evaluate_coverage_student_closed_loop(
        CoverageExplorationActor(),
        scene_ids=(620, 621),
        maximum_steps=2,
    )

    assert report["schema"] == "flightrl.coverage.student_closed_loop.v1"
    assert set(report["modes"]) == {"clean", "frozen", "history_permuted"}
    assert report["modes"]["clean"]["camera_history"] == "own_current"
    assert report["modes"]["frozen"]["camera_history"] == "own_first_frame_repeated"
    assert report["modes"]["history_permuted"]["camera_history"] == (
        "fixed_cyclic_scene_donor_current_frames"
    )
    assert all(mode["episodes"] == 2 for mode in report["modes"].values())
    assert all(mode["collision_rate"] == 0.0 for mode in report["modes"].values())
    assert all(
        "minimum_front_clearance_m" in episode
        for mode in report["modes"].values()
        for episode in mode["episode_results"]
    )
    assert all(
        mode["boundary_violation_rate"] == 0.0
        for mode in report["modes"].values()
    )
    assert "clean_has_no_collision_or_boundary" in report["closed_loop_gate"][
        "checks"
    ]
    assert "clean_encounters_obstacle_challenge" in report["closed_loop_gate"][
        "checks"
    ]
    assert report["closed_loop_gate"]["checks"][
        "clean_encounters_obstacle_challenge"
    ] is False
    assert report["held_out_scene_ids"] == [620, 621]
    assert report["closed_loop_gate"]["thresholds"] == {
        "minimum_clean_coverage_margin": 0.05,
        "minimum_clean_path_length_m": 0.5,
        "maximum_obstacle_challenge_clearance_m": 0.85,
        "required_obstacle_challenge_rate": 1.0,
    }
    assert report["training_authority"] is False
    assert report["generalization_authority"] is False
    assert report["deployment_authority"] is False
    assert report["shadow_authority"] is False
    assert report["flight_authority"] is False
