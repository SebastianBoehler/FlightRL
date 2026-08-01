from __future__ import annotations

import pytest

from flightrl.puffer4_door_eval_stats import marginal_group_evidence


def _group_metrics() -> dict[str, float]:
    return {
        "n": 100.0,
        "success_rate": 0.6,
        "scene_group_schema_version": 1.0,
        "layout_family_1_episode_fraction": 0.2,
        "layout_family_1_success_fraction": 0.1,
        "layout_family_2_episode_fraction": 0.3,
        "layout_family_2_success_fraction": 0.15,
        "layout_family_3_episode_fraction": 0.1,
        "layout_family_3_success_fraction": 0.03,
        "door_face_1_episode_fraction": 0.25,
        "door_face_1_success_fraction": 0.1,
        "door_face_2_episode_fraction": 0.25,
        "door_face_2_success_fraction": 0.2,
        "door_face_3_episode_fraction": 0.25,
        "door_face_3_success_fraction": 0.15,
        "low_light_episode_fraction": 0.0,
        "low_light_success_fraction": 0.0,
        "obstacle_episode_fraction": 0.1,
        "obstacle_success_fraction": 0.02,
    }


def test_marginal_groups_reconstruct_complement_and_supported_worst() -> None:
    report = marginal_group_evidence(_group_metrics())

    layout_zero = report["dimensions"]["layout_family"][0]
    assert layout_zero == {
        "category": 0,
        "support": 40,
        "successes": 32,
        "conditional_success_rate": 0.8,
    }
    assert report["dimensions"]["low_light"][1][
        "conditional_success_rate"
    ] is None
    assert report["worst_marginal_group"] == {
        "scope": "marginal_not_joint",
        "dimension": "obstacle",
        "category": 1,
        "support": 10,
        "successes": 2,
        "conditional_success_rate": 0.2,
    }


def test_empty_obstacle_group_is_reported_without_inventing_a_rate() -> None:
    metrics = _group_metrics()
    metrics["obstacle_episode_fraction"] = 0.0
    metrics["obstacle_success_fraction"] = 0.0

    obstacle = marginal_group_evidence(metrics)["dimensions"]["obstacle"]

    assert obstacle[0]["support"] == 100
    assert obstacle[1]["support"] == 0
    assert obstacle[1]["successes"] == 0
    assert obstacle[1]["conditional_success_rate"] is None


def test_marginal_groups_are_unavailable_without_native_schema() -> None:
    assert marginal_group_evidence({"n": 10.0, "success_rate": 0.5}) == {
        "status": "unavailable",
        "reason": "scene_group_schema_version_missing",
    }


def test_marginal_groups_reject_successes_above_support() -> None:
    metrics = _group_metrics()
    metrics["obstacle_episode_fraction"] = 0.1
    metrics["obstacle_success_fraction"] = 0.2

    with pytest.raises(ValueError, match="exceed support"):
        marginal_group_evidence(metrics)
