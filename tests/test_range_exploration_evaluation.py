from __future__ import annotations

import numpy as np
import pytest
import torch

from flightrl.exploration.range_evaluation import (
    classical_frontier_action,
    evaluate_range_candidate,
)
from flightrl.exploration.range_challenge_evaluation import (
    range_obstacle_challenge_cases,
)
from flightrl.exploration.range_policy import RangeExplorationActorCritic


def _frontier_observation(*, column: int) -> np.ndarray:
    observation = np.zeros(4106, dtype=np.float32)
    map_value = observation[:4096].reshape(4, 32, 32)
    map_value[0, 16, 16] = 1.0
    map_value[1, 12:17, 16] = 1.0
    map_value[3, 12, column] = 1.0
    observation[4096:4100] = 0.8
    observation[4100:4104] = 1.0
    return observation


def test_classical_baseline_can_select_direction_without_entering_actor_contract() -> None:
    left = classical_frontier_action(_frontier_observation(column=12))
    right = classical_frontier_action(_frontier_observation(column=20))

    assert left[1] > 0.0
    assert right[1] < 0.0
    assert left[0] == right[0]


def test_untrained_candidate_produces_honest_failed_causal_report() -> None:
    torch.manual_seed(601)
    model = RangeExplorationActorCritic(hidden_size=64)

    report = evaluate_range_candidate(model, seeds=(601, 602), horizon=20)

    assert report["schema"] == "flightrl.range_exploration.evaluation.v5"
    assert set(report["modes"]) == {
        "clean",
        "range_masked",
        "map_masked",
        "stress",
    }
    assert set(report["baselines"]) == {"stationary_scan", "classical_frontier"}
    assert len(report["obstacle_challenge"]["episodes"]) == 4
    assert "dedicated_obstacle_challenge" in report["checks"]
    assert "all_clean_episodes_have_front_challenge" not in report["checks"]
    assert "mirrored_frontier_direction" not in report["checks"]
    assert "mirrored_frontier_direction" in report["counterfactuals"]
    clean = report["modes"]["clean"]
    stationary = report["baselines"]["stationary_scan"]
    assert clean["mean_final_objective"] == pytest.approx(
        0.35 * clean["mean_final_visited"]
        + 0.65 * clean["mean_final_coverage"]
    )
    assert stationary["mean_final_objective"] == pytest.approx(
        0.35 * stationary["mean_final_visited"]
        + 0.65 * stationary["mean_final_coverage"]
    )
    assert "safety_terminal_rate" in clean
    assert report["simulation_gate_passed"] is False
    assert report["authority"] == {
        "training": False,
        "shadow": False,
        "deployment": False,
        "flight": False,
    }
    assert report["actor_observation_contains_truth"] is False
    assert report["actor_receives_selected_frontier"] is False


def test_dedicated_obstacle_cases_start_close_but_collision_free() -> None:
    cases = range_obstacle_challenge_cases()
    reference_ranges = None

    assert len(cases) == 4
    for case in cases:
        ranges, validity = case.world.horizontal_ranges(case.initial_pose)
        assert case.world.collides(case.initial_pose.x_m, case.initial_pose.y_m) is False
        assert validity[0] == 1.0
        assert 0.60 <= ranges[0] <= 0.66
        if reference_ranges is None:
            reference_ranges = ranges
        else:
            np.testing.assert_allclose(ranges, reference_ranges, atol=0.011)
