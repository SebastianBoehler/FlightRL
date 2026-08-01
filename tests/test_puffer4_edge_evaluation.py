from __future__ import annotations

import pytest
import torch

import flightrl.puffer4_edge_evaluation as edge_evaluation
from flightrl.puffer4_edge_dataset import EDGE_STUDENT_OBSERVATION_DIM
from flightrl.puffer4_edge_evaluation_gate import (
    EDGE_EVALUATION_PROFILES,
    collision_rate_upper_95,
    edge_student_gate,
)
from flightrl.puffer4_edge_schema import EDGE_ACTION_DIM, EDGE_OBSERVATION_DIM


PROFILE = EDGE_EVALUATION_PROFILES[1][3]


def test_evaluation_observation_views_follow_contract_dimensions() -> None:
    observations = torch.arange(EDGE_STUDENT_OBSERVATION_DIM).reshape(1, -1)

    model, action, grounding = edge_evaluation._evaluation_observation_views(
        observations
    )

    assert model.shape == (1, EDGE_OBSERVATION_DIM)
    assert action.shape == (1, EDGE_ACTION_DIM)
    assert grounding.shape == (1, 4)
    assert torch.equal(torch.cat((model, action, grounding), dim=1), observations)


def _passing_metrics() -> dict[str, float]:
    episodes = 256.0
    collision_rate = 2.0 / episodes
    metrics = {
        "success_rate": 244.0 / episodes,
        "collision_rate": collision_rate,
        "collision_rate_upper_95": collision_rate_upper_95(
            collision_rate,
            episodes,
        ),
        "outside_fov_success_rate": 58.0 / 64.0,
        "outside_fov_success_fraction": 58.0 / episodes,
        "outside_fov_episode_fraction": 64.0 / episodes,
        "outside_fov_episodes": 64.0,
        "episodes": episodes,
        "action_rmse": 0.10,
        "door_action_rmse": 0.10,
        "reset_action_rmse": 0.15,
        "reset_door_action_rmse": 0.20,
        "reset_samples": 256.0,
        "lateral_action_abs_mean": 0.02,
        "vertical_action_abs_mean": 0.02,
        "lateral_action_abs_max": 0.10,
        "vertical_action_abs_max": 0.10,
        "grounding_visibility_precision": 0.90,
        "grounding_visibility_recall": 0.90,
        "grounding_visible_box_mae": 0.05,
        "grounding_visible_samples": 10_000.0,
        "grounding_absent_samples": 10_000.0,
        "hidden_min": 0.0,
        "hidden_max": 5.5,
        "low_light_episode_fraction": 0.25,
        "low_light_success_fraction": 60.0 / episodes,
        "obstacle_episode_fraction": 0.50,
        "obstacle_success_fraction": 120.0 / episodes,
    }
    for group in ("layout_family", "door_face"):
        for index in range(1, 4):
            metrics[f"{group}_{index}_episode_fraction"] = 0.25
            metrics[f"{group}_{index}_success_fraction"] = 61.0 / episodes
    return metrics


def test_edge_student_gate_requires_mission_and_four_axis_health() -> None:
    gate = edge_student_gate(_passing_metrics(), profile=PROFILE)

    assert gate["passed"] is True
    assert gate["failures"] == []


def test_edge_student_gate_rejects_missing_nonfinite_or_truthy_values() -> None:
    metrics = _passing_metrics()
    metrics["success_rate"] = float("nan")
    metrics["collision_rate"] = True
    metrics.pop("episodes")

    gate = edge_student_gate(metrics, profile=PROFILE)

    assert gate["passed"] is False
    assert set((
        "success_rate", "collision_rate", "collision_rate_upper_95",
        "outside_fov_episodes", "episodes",
    )) <= set(gate["failures"])


def test_edge_student_gate_rejects_impossible_metric_ranges() -> None:
    metrics = _passing_metrics()
    metrics.update(
        {
            "success_rate": 2.0,
            "collision_rate": -1.0,
            "outside_fov_episodes": 300.0,
            "lateral_action_abs_mean": -0.1,
            "grounding_visibility_precision": 1.1,
            "hidden_max": 6.1,
        }
    )

    gate = edge_student_gate(metrics, profile=PROFILE)

    assert gate["passed"] is False
    assert set((
        "success_rate",
        "collision_rate",
        "collision_rate_upper_95",
        "outside_fov_episodes",
        "lateral_action_abs_mean",
        "grounding_visibility_precision",
        "hidden_max",
    )) <= set(gate["failures"])


def test_edge_student_gate_rejects_failure_isolated_to_derived_group_zero() -> None:
    metrics = _passing_metrics()
    metrics["success_rate"] = 231.0 / metrics["episodes"]
    for index in range(1, 4):
        metrics[f"door_face_{index}_success_fraction"] = 56.0 / metrics["episodes"]

    gate = edge_student_gate(metrics, profile=PROFILE)

    assert gate["passed"] is False
    assert gate["failures"] == ["layout_family_0_success_rate"]


def test_edge_student_gate_requires_conditional_subgroup_coverage_and_success() -> None:
    metrics = _passing_metrics()
    metrics["low_light_episode_fraction"] = 15.0 / metrics["episodes"]
    metrics["low_light_success_fraction"] = 12.0 / metrics["episodes"]
    metrics["obstacle_success_fraction"] = 107.0 / metrics["episodes"]

    gate = edge_student_gate(metrics, profile=PROFILE)

    assert gate["passed"] is False
    assert gate["failures"] == ["low_light_episodes", "obstacle_success_rate"]


def test_edge_student_clean_gate_does_not_invent_optional_subgroup_samples() -> None:
    metrics = _passing_metrics()
    metrics["low_light_episode_fraction"] = 0.0
    metrics["low_light_success_fraction"] = 0.0
    metrics["obstacle_episode_fraction"] = 0.0
    metrics["obstacle_success_fraction"] = 0.0

    gate = edge_student_gate(metrics, profile=EDGE_EVALUATION_PROFILES[0][3])

    assert gate["passed"] is True
    assert "low_light_episodes" not in gate["checks"]
    assert "obstacle_episodes" not in gate["checks"]


def test_edge_student_gate_requires_outside_fov_and_grounding_evidence() -> None:
    metrics = _passing_metrics()
    metrics["outside_fov_episodes"] = 31.0
    metrics["grounding_visibility_recall"] = 0.79
    metrics["reset_action_rmse"] = 0.31

    gate = edge_student_gate(metrics, profile=PROFILE)

    assert gate["passed"] is False
    assert gate["failures"] == [
        "outside_fov_episodes",
        "reset_action_rmse",
        "grounding_visibility_recall",
    ]


def test_edge_student_gate_rejects_rare_unsupported_axis_spikes() -> None:
    metrics = _passing_metrics()
    metrics["lateral_action_abs_max"] = 0.251
    metrics["vertical_action_abs_max"] = 1.0

    gate = edge_student_gate(metrics, profile=PROFILE)

    assert gate["passed"] is False
    assert gate["failures"] == [
        "lateral_action_abs_max",
        "vertical_action_abs_max",
    ]


def test_edge_student_gate_recomputes_collision_confidence_bound() -> None:
    metrics = _passing_metrics()
    metrics["collision_rate_upper_95"] = 0.0

    gate = edge_student_gate(metrics, profile=PROFILE)

    assert gate["passed"] is False
    assert gate["failures"] == ["collision_rate_upper_95"]


def test_edge_student_gate_rejects_fractional_episode_counts() -> None:
    metrics = _passing_metrics()
    metrics["episodes"] = 768_000.4
    metrics["collision_rate"] = 0.0
    metrics["collision_rate_upper_95"] = 0.0

    gate = edge_student_gate(metrics, profile=PROFILE)

    assert gate["passed"] is False
    assert "episodes" in gate["failures"]
    assert "collision_rate_upper_95" in gate["failures"]
    with pytest.raises(ValueError, match="exact episode count"):
        collision_rate_upper_95(0.0, 768_000.4)


def test_edge_student_gate_rejects_fraction_no_native_run_can_emit() -> None:
    metrics = _passing_metrics()
    metrics["episodes"] = 768_000.0
    metrics["collision_rate_upper_95"] = collision_rate_upper_95(
        metrics["collision_rate"], metrics["episodes"]
    )
    metrics["outside_fov_episodes"] = 192_000.0
    metrics["low_light_episode_fraction"] = 200_000.1 / metrics["episodes"]
    metrics["low_light_success_fraction"] = 180_000.05 / metrics["episodes"]

    gate = edge_student_gate(metrics, profile=PROFILE)

    assert gate["passed"] is False
    assert "low_light_episodes" in gate["failures"]
    with pytest.raises(ValueError, match="exact episode count"):
        collision_rate_upper_95(6_000.005 / 768_000.0, 768_000.0)


def test_closed_loop_totals_measure_reset_and_grounding_denominators() -> None:
    totals = edge_evaluation._empty_totals()
    action = torch.tensor(((0.0, -0.25, 0.10, 0.0), (0.0, 0.05, -0.20, 0.0)))
    target_action = torch.ones(2, 4)
    grounding = torch.tensor(((1.0, 0.2, -0.1, 0.4), (0.0, 0.0, 0.0, 0.0)))

    edge_evaluation._accumulate(
        totals,
        action,
        grounding,
        torch.zeros(2, 48),
        target_action,
        grounding,
        torch.tensor((True, False)),
    )
    metrics = edge_evaluation._finish_totals(totals, samples=2)

    assert metrics["action_rmse"] == pytest.approx(
        float(((action - target_action).square().mean()).sqrt())
    )
    assert metrics["door_action_rmse"] == pytest.approx(1.0)
    assert metrics["reset_action_rmse"] == pytest.approx(
        float(((action[0] - target_action[0]).square().mean()).sqrt())
    )
    assert metrics["reset_door_action_rmse"] == pytest.approx(1.0)
    assert metrics["reset_samples"] == 1.0
    assert metrics["grounding_visibility_precision"] == 1.0
    assert metrics["grounding_visibility_recall"] == 1.0
    assert metrics["grounding_visible_samples"] == 1.0
    assert metrics["grounding_absent_samples"] == 1.0
    assert metrics["grounding_visible_box_mae"] == 0.0
    assert metrics["lateral_action_abs_mean"] == pytest.approx(0.15)
    assert metrics["vertical_action_abs_mean"] == pytest.approx(0.15)
    assert metrics["lateral_action_abs_max"] == pytest.approx(0.25)
    assert metrics["vertical_action_abs_max"] == pytest.approx(0.20)
