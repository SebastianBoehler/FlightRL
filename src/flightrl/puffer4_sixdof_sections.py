from __future__ import annotations

from .puffer4_config import Puffer4ExportSettings
from .sixdof.curriculum import resolve_reset_profile
from .sixdof.sensor_model import resolve_sensor_profile


TASK_IDS = {"position_yaw": 0, "obstacle_avoidance": 1, "attitude": 2, "circle": 3}
REWARD_MODE_IDS = {"env": 0, "progress": 1, "progress_clearance": 2, "progress_yaw_clearance": 3, "live_clearance": 4}


def build_sixdof_sections(settings: Puffer4ExportSettings) -> dict[str, dict[str, int | float | str]]:
    total_agents = settings.total_agents or 4096
    num_buffers = settings.num_buffers or 8
    hidden_size = settings.policy_hidden_size or 128
    sensor_profile = resolve_sensor_profile(settings.sim_profile)
    return {
        "base": {"env_name": settings.env_name, "checkpoint_interval": 10, "seed": settings.train_seed},
        "vec": {"total_agents": total_agents, "num_buffers": num_buffers, "num_threads": settings.num_threads or num_buffers},
        "env": {
            "seed": settings.train_seed,
            "dt": 0.01,
            "room_x_min": -2.0,
            "room_x_max": 2.0,
            "room_y_min": -2.0,
            "room_y_max": 2.0,
            "room_z_min": 0.0,
            "room_z_max": 2.5,
            "max_range_m": 4.0,
            "mass_kg": 0.036,
            "gravity_m_s2": 9.81,
            "linear_drag": 0.08,
            "rate_tau_s": 0.045,
            "thrust_scale": 0.75,
            "max_rate_roll": 6.0,
            "max_rate_pitch": 6.0,
            "max_rate_yaw": 4.0,
            "motor_tau_s": 0.035,
            **sensor_profile.as_env_values(),
            **reset_profile_values(settings.reset_profile),
            "task_id": resolve_id(settings.task, TASK_IDS, "task"),
            "reward_mode": resolve_id(settings.reward_mode, REWARD_MODE_IDS, "reward_mode"),
        },
        "policy": {"hidden_size": hidden_size, "num_layers": settings.policy_num_layers, "expansion_factor": 1},
        "torch": {"network": "MLP", "encoder": "DefaultEncoder", "decoder": "DefaultDecoder"},
        "train": {
            "gpus": 1,
            "seed": settings.train_seed,
            "total_timesteps": 1048576,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "replay_ratio": 2,
            "clip_coef": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.001,
            "minibatch_size": 8192,
            "horizon": 32,
        },
    }


def reset_profile_values(name: str) -> dict[str, float]:
    profile = resolve_reset_profile(name)
    near_wall = profile.near_wall_clearance_range or (0.0, 0.0)
    return {
        "near_wall_probability": profile.near_wall_probability,
        "near_wall_min_clearance_m": near_wall[0],
        "near_wall_max_clearance_m": near_wall[1],
        "near_wall_yaw_jitter_rad": profile.near_wall_yaw_jitter_rad,
        "reset_z_min": profile.z_range[0],
        "reset_z_max": profile.z_range[1],
        "target_z_min": profile.target_z_range[0],
        "target_z_max": profile.target_z_range[1],
        "target_xy_offset_abs": -1.0 if profile.target_xy_offset_abs is None else profile.target_xy_offset_abs,
        "target_z_offset_abs": -1.0 if profile.target_z_offset_abs is None else profile.target_z_offset_abs,
        "target_yaw_offset_abs": -1.0 if profile.target_yaw_offset_abs is None else profile.target_yaw_offset_abs,
    }


def resolve_id(value: str, choices: dict[str, int], label: str) -> int:
    try:
        return int(value)
    except ValueError:
        pass
    if value not in choices:
        raise ValueError(f"unknown {label} {value!r}; expected one of {sorted(choices)}")
    return choices[value]
