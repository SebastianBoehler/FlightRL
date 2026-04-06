from __future__ import annotations

from .config import FlightConfig, MAX_WAYPOINTS


TASK_MAP = {
    "hover": 0,
    "reach_waypoint": 1,
    "follow_waypoints": 2,
}

ACTION_MAP = {
    "stabilized_planar": 0,
    "motor_pair": 1,
    "motor_quad": 2,
}

RESET_MAP = {
    "deterministic": 0,
    "random_uniform": 1,
}


def build_binding_kwargs(config: FlightConfig) -> dict[str, float | int]:
    task = config.task
    drone = config.drone
    reward = config.reward
    sensors = config.sensors
    randomization = config.domain_randomization
    wind = config.wind
    kwargs: dict[str, float | int] = {
        "dt": config.environment.dt,
        "action_dim": config.action_dim,
        "observation_dim": config.observation_dim,
        "observation_flags": config.observation_flags,
        "state_noise_std": sensors.state_noise_std,
        "imu_noise_std": sensors.imu_noise_std,
        "task_type": TASK_MAP[task.task_type],
        "action_mode": ACTION_MAP[config.environment.action_mode],
        "reset_mode": RESET_MAP[config.environment.reset_mode],
        "max_steps": task.max_steps,
        "sequence_length": min(task.sequence_length, MAX_WAYPOINTS),
        "hover_hold_steps": task.hover_hold_steps,
        "success_radius": task.success_radius,
        "hover_speed_threshold": task.hover_speed_threshold,
        "fixed_start_x": task.fixed_start[0],
        "fixed_start_z": task.fixed_start[1],
        "fixed_target_x": task.fixed_target[0],
        "fixed_target_z": task.fixed_target[1],
        "spawn_x_min": task.spawn_bounds[0],
        "spawn_x_max": task.spawn_bounds[1],
        "spawn_z_min": task.spawn_bounds[2],
        "spawn_z_max": task.spawn_bounds[3],
        "target_x_min": task.target_bounds[0],
        "target_x_max": task.target_bounds[1],
        "target_z_min": task.target_bounds[2],
        "target_z_max": task.target_bounds[3],
        "mass": drone.mass,
        "inertia": drone.inertia,
        "arm_length": drone.arm_length,
        "drag": drone.drag,
        "angular_drag": drone.angular_drag,
        "gravity": drone.gravity,
        "hover_thrust": drone.hover_thrust,
        "thrust_gain": drone.thrust_gain,
        "max_total_thrust": drone.max_total_thrust,
        "max_pitch_torque": drone.max_pitch_torque,
        "actuator_tau": drone.actuator_tau,
        "max_velocity": drone.max_velocity,
        "max_pitch_rate": drone.max_pitch_rate,
        "max_pitch_angle": drone.max_pitch_angle,
        "floor_z": drone.floor_z,
        "x_limit": drone.x_limit,
        "z_limit": drone.z_limit,
        "alive_bonus": reward.alive_bonus,
        "distance_penalty": reward.distance_penalty,
        "progress_bonus": reward.progress_bonus,
        "velocity_penalty": reward.velocity_penalty,
        "angular_rate_penalty": reward.angular_rate_penalty,
        "control_penalty": reward.control_penalty,
        "smoothness_penalty": reward.smoothness_penalty,
        "success_bonus": reward.success_bonus,
        "crash_penalty": reward.crash_penalty,
        "out_of_bounds_penalty": reward.out_of_bounds_penalty,
        "randomization_enabled": int(randomization.enabled),
        "mass_scale": randomization.mass_scale,
        "drag_scale": randomization.drag_scale,
        "thrust_scale": randomization.thrust_scale,
        "actuator_tau_scale": randomization.actuator_tau_scale,
        "sensor_noise_scale": randomization.sensor_noise_scale,
        "wind_enabled": int(wind.enabled),
        "wind_steady_x": wind.steady_x,
        "wind_steady_z": wind.steady_z,
        "wind_gust_strength": wind.gust_strength,
        "wind_gust_tau": wind.gust_tau,
    }
    for idx in range(MAX_WAYPOINTS):
        waypoint = task.fixed_waypoints[idx]
        kwargs[f"waypoint_{idx}_x"] = waypoint[0]
        kwargs[f"waypoint_{idx}_z"] = waypoint[1]
    return kwargs
