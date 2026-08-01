from __future__ import annotations

import torch

from flightrl.puffer4_vision_training import (
    visual_simulation_gate,
    visual_teacher_actions,
)


def test_visual_teacher_steers_toward_open_image_half() -> None:
    observations = torch.zeros((3, 3 * 16 * 12 + 6 + 1))
    observations[:, 3 * 16 * 12 + 3] = torch.tensor([0.6, 0.6, 0.2])
    observations[:, -1] = torch.tensor([1.0, -1.0, 1.0])

    actions = visual_teacher_actions(observations)

    assert actions[0, 1] == 1.0
    assert actions[1, 1] == -1.0
    assert actions[2, 1] == 0.0
    assert torch.count_nonzero(actions[:, (0, 2, 3)]) == 0


def test_visual_simulation_gate_requires_robust_camera_dependent_control() -> None:
    evaluation = {
        "obstacle_full_vision": {"success_rate": 0.92, "collision_rate": 0.03},
        "obstacle_masked_vision": {"success_rate": 0.05},
        "clear_full_vision": {
            "success_rate": 0.96,
            "lateral_action_p95": 0.20,
        },
        "nominal_obstacle_full_vision": {"success_rate": 0.98},
    }

    assert visual_simulation_gate(evaluation)["passed"] is True

    evaluation["obstacle_masked_vision"]["success_rate"] = 0.60
    gate = visual_simulation_gate(evaluation)
    assert gate["passed"] is False
    assert gate["failures"] == ["camera_dependence"]
