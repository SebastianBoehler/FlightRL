from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flightrl.mujoco.odometry import OdometryNoiseConfig


@dataclass(frozen=True, slots=True)
class SemanticVisionEnvConfig:
    odometry: OdometryNoiseConfig = OdometryNoiseConfig()
    progress_reward_scale: float = 8.0
    exploration_reward_scale: float = 0.01
    action_penalty_scale: float = 0.005
    collision_penalty: float = 5.0
    success_reward: float = 5.0
    clearance_penalty_scale: float = 1.5
    unsafe_clearance_m: float = 0.45
    progress_after_target_only: bool = False
    success_requires_target_evidence: bool = False
    episode_max_steps: int = 800
    use_range_map_updates: bool = False


def project_semantic_actions(
    actions: np.ndarray,
    *,
    action_mode: str,
    max_yawrate_deg_s: float,
) -> np.ndarray:
    commands = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
    if action_mode != "active_exploration":
        return commands
    commands[:, 0] = np.clip(commands[:, 0], 0.0, 1.0)
    commands[:, 1:3] = 0.0
    yaw_limit = 20.0 / max_yawrate_deg_s
    commands[:, 3] = np.clip(commands[:, 3], -yaw_limit, yaw_limit)
    return commands


def semantic_rewards(
    config: SemanticVisionEnvConfig,
    *,
    previous_distance: np.ndarray,
    distance: np.ndarray,
    target_acquired: np.ndarray,
    new_cells: np.ndarray,
    commands: np.ndarray,
    front_clearance: np.ndarray,
    collisions: np.ndarray,
    success: np.ndarray,
) -> np.ndarray:
    progress_active = (
        target_acquired
        if config.progress_after_target_only
        else np.ones(len(distance), dtype=bool)
    )
    clearance_pressure = np.maximum(
        0.0,
        config.unsafe_clearance_m - front_clearance,
    )
    return (
        config.progress_reward_scale
        * (previous_distance - distance)
        * progress_active
        + config.exploration_reward_scale * new_cells
        - config.action_penalty_scale * np.sum(commands * commands, axis=1)
        - config.clearance_penalty_scale * clearance_pressure
        - config.collision_penalty * collisions
        + config.success_reward * success
    ).astype(np.float32)


def semantic_episode_info(
    done: np.ndarray,
    success: np.ndarray,
    collision: np.ndarray,
    target_acquired: np.ndarray,
    episode_return: np.ndarray,
) -> list[dict[str, float]]:
    if not np.any(done):
        return []
    return [
        {
            "n": float(np.sum(done)),
            "success_rate": float(np.mean(success[done])),
            "collision_rate": float(np.mean(collision[done])),
            "target_discovery_rate": float(np.mean(target_acquired[done])),
            "episode_return": float(np.mean(episode_return[done])),
        }
    ]
