from flightrl.mujoco.semantic_reporting import (
    select_semantic_candidate,
    shadow_gate_passed,
    teacher_gate_passed,
)


def test_active_candidate_selection_rejects_unsafe_policy() -> None:
    evaluations = {
        "unsafe": {
            "full": {
                "collision_rate": 0.1,
                "target_discovery_rate": 1.0,
                "success_rate": 1.0,
                "unsafe_forward_fraction": 0.0,
                "minimum_moving_front_clearance_m": 0.5,
                "clearance_false_safe_fraction": 0.0,
                "max_lateral_vertical_action": 0.0,
            }
        },
        "safe": {
            "full": {
                "collision_rate": 0.0,
                "target_discovery_rate": 0.8,
                "success_rate": 0.6,
                "unsafe_forward_fraction": 0.01,
                "minimum_moving_front_clearance_m": 0.5,
                "clearance_false_safe_fraction": 0.01,
                "max_lateral_vertical_action": 0.0,
            }
        },
    }

    assert select_semantic_candidate(evaluations, active_exploration=True) == "safe"


def test_active_candidate_selection_prefers_lower_risk_when_all_fail_gate() -> None:
    evaluations = {
        "bootstrap": {
            "full": {
                "collision_rate": 0.0625,
                "target_discovery_rate": 0.75,
                "success_rate": 0.0,
                "unsafe_forward_fraction": 0.037,
                "minimum_moving_navigation_clearance_m": 0.086,
                "clearance_false_safe_fraction": 0.124,
                "max_lateral_vertical_action": 0.0,
                "preacquisition_forward_mean": 0.154,
            }
        },
        "puffer_ppo": {
            "full": {
                "collision_rate": 0.0625,
                "target_discovery_rate": 0.75,
                "success_rate": 0.0,
                "unsafe_forward_fraction": 0.042,
                "minimum_moving_navigation_clearance_m": 0.085,
                "clearance_false_safe_fraction": 0.119,
                "max_lateral_vertical_action": 0.0,
                "preacquisition_forward_mean": 0.164,
            }
        },
    }

    assert (
        select_semantic_candidate(evaluations, active_exploration=True)
        == "bootstrap"
    )


def test_active_candidate_selection_uses_horizontal_clearance_when_available() -> None:
    evaluations = {
        "side_collision": {
            "full": {
                "collision_rate": 0.0,
                "target_discovery_rate": 1.0,
                "success_rate": 1.0,
                "unsafe_forward_fraction": 0.0,
                "minimum_moving_front_clearance_m": 2.0,
                "minimum_moving_horizontal_clearance_m": 0.1,
                "clearance_false_safe_fraction": 0.0,
                "max_lateral_vertical_action": 0.0,
            }
        },
        "safe": {
            "full": {
                "collision_rate": 0.0,
                "target_discovery_rate": 0.8,
                "success_rate": 0.6,
                "unsafe_forward_fraction": 0.0,
                "minimum_moving_front_clearance_m": 0.5,
                "minimum_moving_horizontal_clearance_m": 0.5,
                "clearance_false_safe_fraction": 0.0,
                "max_lateral_vertical_action": 0.0,
            }
        },
    }

    assert select_semantic_candidate(evaluations, active_exploration=True) == "safe"


def test_active_shadow_gate_requires_discovery_and_forward_only_projection() -> None:
    passing = {
        "success_rate": 0.5,
        "target_discovery_rate": 0.7,
        "collision_rate": 0.02,
        "unsafe_forward_fraction": 0.02,
        "minimum_moving_front_clearance_m": 0.25,
        "clearance_false_safe_fraction": 0.02,
        "max_lateral_vertical_action": 0.001,
    }

    teacher = {
        "success_rate": 0.8,
        "target_discovery_rate": 0.9,
        "collision_rate": 0.0,
        "unsafe_forward_fraction": 0.0,
        "minimum_moving_front_clearance_m": 0.65,
        "max_lateral_vertical_action": 0.0,
    }

    assert teacher_gate_passed(teacher)
    assert shadow_gate_passed(
        passing,
        {},
        active_exploration=True,
        teacher=teacher,
    )
    assert not shadow_gate_passed(
        {**passing, "target_discovery_rate": 0.69},
        {},
        active_exploration=True,
        teacher=teacher,
    )
    assert not shadow_gate_passed(
        {**passing, "clearance_false_safe_fraction": 0.021},
        {},
        active_exploration=True,
        teacher=teacher,
    )
    assert not shadow_gate_passed(
        passing,
        {},
        active_exploration=True,
        teacher={**teacher, "collision_rate": 0.03},
    )
