from __future__ import annotations

import numpy as np

from flightrl.mujoco.semantic_puffer_driver import SemanticPufferDriver


def evaluate_semantic_teacher(
    driver: SemanticPufferDriver,
    *,
    steps: int,
) -> dict[str, float]:
    driver.reset()
    action_norms = []
    preacquisition_forward = []
    front_clearance_samples = []
    horizontal_clearance_samples = []
    navigation_clearance_samples = []
    moving_front_clearance_samples = []
    moving_horizontal_clearance_samples = []
    moving_navigation_clearance_samples = []
    unsafe_forward_samples = 0
    max_horizontal_speed = 0.0
    max_abs_yawrate = 0.0
    max_lateral_vertical_action = 0.0
    visible_samples = 0
    acquired_samples = 0
    control = driver.driver_env.control
    for _ in range(steps):
        actions = np.ascontiguousarray(driver.expert_actions(), dtype=np.float32)
        observed = driver.target_observed()
        visible = driver.target_visible()
        front_clearance = driver.front_clearance()
        horizontal_clearance = driver.horizontal_clearance()
        navigation_clearance = driver.navigation_clearance()
        physical_horizontal = (
            np.linalg.norm(actions[:, :2], axis=1)
            * control.max_horizontal_speed_m_s
        )
        moving_forward = actions[:, 0] > 0.1
        max_horizontal_speed = max(
            max_horizontal_speed,
            float(np.max(physical_horizontal)),
        )
        max_abs_yawrate = max(
            max_abs_yawrate,
            float(np.max(np.abs(actions[:, 3]))) * control.max_yawrate_deg_s,
        )
        max_lateral_vertical_action = max(
            max_lateral_vertical_action,
            float(np.max(np.abs(actions[:, 1:3]))),
        )
        preacquisition_forward.extend(actions[~observed, 0].tolist())
        front_clearance_samples.extend(front_clearance.tolist())
        horizontal_clearance_samples.extend(horizontal_clearance.tolist())
        navigation_clearance_samples.extend(navigation_clearance.tolist())
        moving_front_clearance_samples.extend(front_clearance[moving_forward].tolist())
        moving_horizontal_clearance_samples.extend(
            horizontal_clearance[moving_forward].tolist()
        )
        moving_navigation_clearance_samples.extend(
            navigation_clearance[moving_forward].tolist()
        )
        unsafe_forward_samples += int(
            np.sum(moving_forward & (navigation_clearance < 0.45))
        )
        visible_samples += int(np.sum(visible))
        acquired_samples += int(np.sum(observed))
        action_norms.extend(np.linalg.norm(actions, axis=1).tolist())
        driver.teacher_step(actions)
    sample_count = max(1, steps * driver.total_agents)
    result = driver.log()
    result.update(
        {
            "mean_action_l2": float(np.mean(action_norms)),
            "steps": float(steps * driver.total_agents),
            "max_horizontal_speed_m_s": max_horizontal_speed,
            "max_abs_yawrate_deg_s": max_abs_yawrate,
            "target_visible_fraction": visible_samples / sample_count,
            "target_acquired_fraction": acquired_samples / sample_count,
            "preacquisition_forward_mean": (
                float(np.mean(preacquisition_forward))
                if preacquisition_forward
                else 0.0
            ),
            "minimum_front_clearance_m": min(
                front_clearance_samples,
                default=4.0,
            ),
            "minimum_horizontal_clearance_m": min(
                horizontal_clearance_samples,
                default=4.0,
            ),
            "minimum_navigation_clearance_m": min(
                navigation_clearance_samples,
                default=4.0,
            ),
            "minimum_moving_front_clearance_m": min(
                moving_front_clearance_samples,
                default=4.0,
            ),
            "minimum_moving_horizontal_clearance_m": min(
                moving_horizontal_clearance_samples,
                default=4.0,
            ),
            "minimum_moving_navigation_clearance_m": min(
                moving_navigation_clearance_samples,
                default=4.0,
            ),
            "unsafe_forward_fraction": unsafe_forward_samples / sample_count,
            "max_lateral_vertical_action": max_lateral_vertical_action,
        }
    )
    return {key: float(value) for key, value in result.items()}
