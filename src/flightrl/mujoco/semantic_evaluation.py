from __future__ import annotations

import numpy as np
import torch

from flightrl.mujoco.semantic_puffer_driver import SemanticPufferDriver
from flightrl.mujoco.semantic_vision_policy import SemanticVisionPolicy
from flightrl.navigation.spatial_memory import MAP_CHANNELS

@torch.no_grad()
def evaluate_semantic_policy(
    policy: SemanticVisionPolicy,
    driver: SemanticPufferDriver,
    *,
    steps: int,
    mode: str,
) -> dict[str, float]:
    driver.reset()
    policy.eval()
    state = policy.initial_state(driver.total_agents, "cpu")
    action_norms = []
    preacquisition_speeds = []
    preacquisition_forward = []
    visible_yaw_errors = []
    visible_yaw_signs = []
    visible_yaw_rates = []
    max_horizontal_speed = 0.0
    max_abs_yawrate = 0.0
    visible_samples = 0
    acquired_samples = 0
    front_clearance_samples = []
    corridor_clearance_samples = []
    horizontal_clearance_samples = []
    navigation_clearance_samples = []
    moving_front_clearance_samples = []
    moving_corridor_clearance_samples = []
    moving_horizontal_clearance_samples = []
    moving_navigation_clearance_samples = []
    unsafe_forward_samples = 0
    clearance_errors = []
    clearance_false_safe_samples = 0
    max_lateral_vertical_action = 0.0
    target_channel = MAP_CHANNELS.index("target_evidence")
    control = driver.driver_env.control
    for _ in range(steps):
        observations = torch.from_numpy(driver.observations.copy())
        _apply_ablation(observations, driver, mode, target_channel)
        expert = driver.expert_actions()
        observed = driver.target_observed()
        visible = driver.target_visible()
        distribution, _, state, predicted_clearance_tensor, _predicted_risk = (
            policy.forward_eval_with_aux(observations, state)
        )
        actions = distribution.mean.clamp(-1.0, 1.0).contiguous()
        action_values = actions.numpy()
        physical_horizontal = (
            np.linalg.norm(action_values[:, :2], axis=1)
            * control.max_horizontal_speed_m_s
        )
        max_horizontal_speed = max(
            max_horizontal_speed,
            float(np.max(physical_horizontal)),
        )
        max_abs_yawrate = max(
            max_abs_yawrate,
            float(np.max(np.abs(action_values[:, 3]))) * control.max_yawrate_deg_s,
        )
        preacquisition_speeds.extend(physical_horizontal[~observed].tolist())
        preacquisition_forward.extend(action_values[~observed, 0].tolist())
        front_clearance = driver.front_clearance()
        corridor_clearance = driver.action_corridor_clearance()
        horizontal_clearance = driver.horizontal_clearance()
        navigation_clearance = driver.navigation_clearance()
        if predicted_clearance_tensor is not None:
            predicted_clearance = predicted_clearance_tensor.squeeze(1).numpy()
            clearance_errors.extend(
                np.abs(predicted_clearance - corridor_clearance).tolist()
            )
            clearance_false_safe_samples += int(
                np.sum(
                    (corridor_clearance < 0.45)
                    & (predicted_clearance > 0.65)
                )
            )
        front_clearance_samples.extend(front_clearance.tolist())
        corridor_clearance_samples.extend(corridor_clearance.tolist())
        horizontal_clearance_samples.extend(horizontal_clearance.tolist())
        navigation_clearance_samples.extend(navigation_clearance.tolist())
        moving_forward = action_values[:, 0] > 0.1
        moving_front_clearance_samples.extend(front_clearance[moving_forward].tolist())
        moving_corridor_clearance_samples.extend(
            corridor_clearance[moving_forward].tolist()
        )
        moving_horizontal_clearance_samples.extend(
            horizontal_clearance[moving_forward].tolist()
        )
        moving_navigation_clearance_samples.extend(
            navigation_clearance[moving_forward].tolist()
        )
        unsafe_forward_samples += int(
            np.sum(moving_forward & (navigation_clearance < 0.45))
        )
        max_lateral_vertical_action = max(
            max_lateral_vertical_action,
            float(np.max(np.abs(action_values[:, 1:3]))),
        )
        _collect_visible_yaw_metrics(
            action_values,
            expert,
            visible,
            control.max_yawrate_deg_s,
            visible_yaw_errors,
            visible_yaw_signs,
            visible_yaw_rates,
        )
        visible_samples += int(np.sum(visible))
        acquired_samples += int(np.sum(observed))
        action_norms.extend(np.linalg.norm(action_values, axis=1).tolist())
        driver.cpu_step(np.ascontiguousarray(action_values).ctypes.data)
        alive = torch.from_numpy(1.0 - driver.terminals).reshape(1, -1, 1)
        state = tuple(value * alive for value in state)
    result = driver.log()
    result.update(
        {
            "mean_action_l2": float(np.mean(action_norms)),
            "steps": float(steps * driver.total_agents),
            "preacquisition_horizontal_p95_m_s": _percentile(
                preacquisition_speeds,
                95,
            ),
            "max_horizontal_speed_m_s": max_horizontal_speed,
            "max_abs_yawrate_deg_s": max_abs_yawrate,
            "visible_yaw_mae_deg_s": float(np.mean(visible_yaw_errors))
            if visible_yaw_errors
            else 0.0,
            "visible_yaw_sign_accuracy": float(np.mean(visible_yaw_signs))
            if visible_yaw_signs
            else 0.0,
            "visible_abs_yawrate_p95_deg_s": _percentile(visible_yaw_rates, 95),
            "target_visible_fraction": visible_samples
            / max(1, steps * driver.total_agents),
            "target_acquired_fraction": acquired_samples
            / max(1, steps * driver.total_agents),
            "preacquisition_forward_mean": float(np.mean(preacquisition_forward))
            if preacquisition_forward
            else 0.0,
            "minimum_front_clearance_m": min(
                front_clearance_samples,
                default=4.0,
            ),
            "minimum_action_corridor_clearance_m": min(
                corridor_clearance_samples,
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
            "minimum_moving_action_corridor_clearance_m": min(
                moving_corridor_clearance_samples,
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
            "unsafe_forward_fraction": unsafe_forward_samples
            / max(1, steps * driver.total_agents),
            "clearance_mae_m": float(np.mean(clearance_errors))
            if clearance_errors
            else 0.0,
            "clearance_false_safe_fraction": clearance_false_safe_samples
            / max(1, steps * driver.total_agents),
            "max_lateral_vertical_action": max_lateral_vertical_action,
        }
    )
    return {key: float(value) for key, value in result.items()}


def _apply_ablation(
    observations: torch.Tensor,
    driver: SemanticPufferDriver,
    mode: str,
    target_channel: int,
) -> None:
    layout = driver.driver_env.layout
    if mode == "vision_masked":
        observations[:, layout.vision_slice] = 0.0
    elif mode == "temporal_masked":
        vision = observations[:, layout.vision_slice].reshape(
            driver.total_agents,
            *driver.driver_env.vision_config.shape,
        )
        vision[:, 1:] = 0.0
    elif mode == "target_map_masked":
        maps = observations[:, layout.map_slice].reshape(
            driver.total_agents,
            *driver.driver_env.memory_config.shape,
        )
        maps[:, target_channel] = 0.0
    elif mode == "command_rotated":
        commands = observations[:, layout.command_slice].clone()
        observations[:, layout.command_slice] = commands.roll(1, dims=1)
    elif mode != "full":
        raise ValueError(f"unknown evaluation mode {mode!r}")


def _collect_visible_yaw_metrics(
    actions: np.ndarray,
    expert: np.ndarray,
    visible: np.ndarray,
    yawrate_scale: float,
    errors: list[float],
    signs: list[float],
    rates: list[float],
) -> None:
    for predicted, expected in zip(
        actions[visible, 3],
        expert[visible, 3],
        strict=True,
    ):
        errors.append(float(abs(predicted - expected) * yawrate_scale))
        rates.append(float(abs(predicted) * yawrate_scale))
        signs.append(
            float(
                abs(expected) < 0.05
                and abs(predicted) < 0.10
                or np.sign(predicted) == np.sign(expected)
            )
        )


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0
