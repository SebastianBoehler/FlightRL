from __future__ import annotations

from dataclasses import replace

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_contract import PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
)
from flightrl.puffer4_door_mission import FIXED_DOOR_MISSION_METRIC_V1
from flightrl.puffer4_sixdof_sections import build_sixdof_sections


def build_fixed_door_teacher_sections(
    settings: Puffer4ExportSettings,
) -> dict[str, dict[str, int | float | str]]:
    resolved = replace(
        settings,
        total_agents=settings.total_agents or 256,
        task="position_yaw",
        reward_mode="progress",
    )
    sections = build_sixdof_sections(resolved)
    action = PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT
    sections["env"].update(
        {
            "physics_substeps": 2,
            # The edge-v3 0.25 m/s envelope needs a 40 s mission budget for
            # the longest room-diagonal plus a full outside-FOV search.
            "max_episode_steps": 2600,
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
    sections["env"].update(FIXED_DOOR_MISSION_METRIC_V1.env_values())
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT.apply_to_env(sections["env"])
    sections["base"]["env_name"] = settings.env_name
    sections["train"]["total_timesteps"] = 0
    return sections
