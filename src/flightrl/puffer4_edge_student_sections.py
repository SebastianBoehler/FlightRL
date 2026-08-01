from __future__ import annotations

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_sections import build_fixed_door_teacher_sections
from flightrl.puffer4_edge_schema import ACTION_SPECS


def build_edge_student_sections(
    settings: Puffer4ExportSettings,
) -> dict[str, dict[str, int | float | str]]:
    sections = build_fixed_door_teacher_sections(settings)
    action_scales = {
        name: scale for name, _unit, scale, _frame in ACTION_SPECS
    }
    sections["env"]["max_horizontal_speed_m_s"] = action_scales["vx"]
    sections["env"]["max_vertical_speed_m_s"] = action_scales["vz"]
    sections["env"]["max_yawrate_deg_s"] = action_scales["yaw_rate"]
    sections["policy"]["hidden_size"] = settings.policy_hidden_size or 48
    sections["train"]["total_timesteps"] = 0
    return sections
