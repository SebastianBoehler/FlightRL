from __future__ import annotations

from dataclasses import replace

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_contract import CORRECTED_DOOR_ACTION_CONTRACT
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_sixdof_sections import build_sixdof_sections


def build_fixed_door_sections(
    settings: Puffer4ExportSettings,
) -> dict[str, dict[str, int | float | str]]:
    resolved = replace(
        settings,
        total_agents=settings.total_agents or 256,
        task="position_yaw",
        reward_mode="progress",
    )
    sections = build_sixdof_sections(resolved)
    action = CORRECTED_DOOR_ACTION_CONTRACT
    sections["env"].update(
        {
            "physics_substeps": 2,
            "max_episode_steps": 1300,
            "success_radius_m": 0.80,
            "max_vertical_speed_m_s": 0.10,
            "velocity_gain": 3.0,
            "attitude_gain": 6.0,
            "vertical_gain": 2.0,
            "camera_mean_min": 18.0,
            "camera_mean_max": 110.0,
            "appearance_seed": 2_003,
            "domain_randomization": 1.0,
            "layout_diversity": 0.0,
            "camera_randomization": 0.0,
            "obstacle_probability": 0.50,
            "camera_mask": 0.0,
        }
    )
    action.apply_to_env(sections["env"])
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT.apply_to_env(sections["env"])
    sections["policy"].update(
        {
            "hidden_size": settings.policy_hidden_size or 96,
            "num_layers": 1,
        }
    )
    sections["torch"].update(
        {
            "network": "MinGRU",
            "encoder": "FlightRLDoorEncoder",
        }
    )
    sections["train"].update(
        {
            "total_timesteps": 8_388_608,
            "learning_rate": 0.001,
            "minibatch_size": 4096,
            "horizon": 64,
            "replay_ratio": 4,
            "ent_coef": 0.002,
        }
    )
    return sections
