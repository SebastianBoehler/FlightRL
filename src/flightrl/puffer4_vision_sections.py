from __future__ import annotations

from dataclasses import replace

from .puffer4_config import Puffer4ExportSettings
from .puffer4_sixdof_sections import build_sixdof_sections


def build_visual_navigation_sections(
    settings: Puffer4ExportSettings,
) -> dict[str, dict[str, int | float | str]]:
    resolved = replace(
        settings,
        total_agents=settings.total_agents or 128,
        task="position_yaw",
        reward_mode="progress",
    )
    sections = build_sixdof_sections(resolved)
    sections["env"].update(
        {
            "control_dt": 1.0 / 65.0,
            "physics_substeps": 2,
            "max_episode_steps": 520,
            "success_radius_m": 0.16,
            "max_horizontal_speed_m_s": 0.45,
            "max_vertical_speed_m_s": 0.10,
            "velocity_gain": 3.0,
            "attitude_gain": 6.0,
            "vertical_gain": 2.0,
            "camera_mean_min": 35.0,
            "camera_mean_max": 90.0,
            "obstacle_probability": 0.75,
            "navigation_residual_scale": 0.6,
            "waypoint_slowdown_distance_m": 0.55,
        }
    )
    sections["policy"].update({"hidden_size": settings.policy_hidden_size or 128, "num_layers": 1})
    sections["torch"].update({"network": "MinGRU", "encoder": "FlightRLVisionEncoder"})
    sections["train"].update(
        {
            "total_timesteps": 262144,
            "learning_rate": 0.003,
            "minibatch_size": 2048,
            "horizon": 16,
            "replay_ratio": 2,
        }
    )
    return sections
