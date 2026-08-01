from __future__ import annotations

from dataclasses import replace

from .puffer4_config import Puffer4ExportSettings
from .puffer4_sixdof_sections import build_sixdof_sections


MAX_HORIZONTAL_SPEED_M_S = 0.45
NAVIGATION_RESIDUAL_SCALE = 0.60


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
            "max_episode_steps": 650,
            "success_radius_m": 0.16,
            "max_horizontal_speed_m_s": MAX_HORIZONTAL_SPEED_M_S,
            "max_vertical_speed_m_s": 0.10,
            "velocity_gain": 3.0,
            "attitude_gain": 6.0,
            "vertical_gain": 2.0,
            "camera_mean_min": 18.0,
            "camera_mean_max": 110.0,
            "domain_randomization": 0.0,
            "obstacle_probability": 0.75,
            "navigation_residual_scale": NAVIGATION_RESIDUAL_SCALE,
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
