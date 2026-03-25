from __future__ import annotations

import gymnasium
import numpy as np
import pufferlib

from . import _binding
from .config import FlightConfig, MAX_WAYPOINTS
from .renderer import DroneFrame, FlightRenderer


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


class DronePlanarEnv(pufferlib.PufferEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: FlightConfig,
        num_envs: int | None = None,
        buf=None,
        seed: int = 0,
        emit_logs: bool = True,
        render_mode: str | None = None,
    ):
        self.config = config
        env_count = num_envs or config.environment.num_envs
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.observation_dim,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.action_dim,),
            dtype=np.float32,
        )
        self.num_agents = env_count
        self.render_mode = render_mode
        self._tick = 0
        self._report_interval = config.logging.report_interval
        self._emit_logs = emit_logs
        self._handles: list[int] = []
        self._renderer: FlightRenderer | None = None
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"unsupported render mode: {render_mode}")

        super().__init__(buf)
        self.actions = self.actions.astype(np.float32, copy=False)
        kwargs = self._binding_kwargs()
        for env_idx in range(env_count):
            handle = _binding.env_init(
                self.observations[env_idx : env_idx + 1],
                self.actions[env_idx : env_idx + 1],
                self.rewards[env_idx : env_idx + 1],
                self.terminals[env_idx : env_idx + 1],
                self.truncations[env_idx : env_idx + 1],
                seed + env_idx,
                **kwargs,
            )
            self._handles.append(handle)
        self._vec_handle = _binding.vectorize(*self._handles)

    def _binding_kwargs(self) -> dict[str, float | int]:
        task = self.config.task
        drone = self.config.drone
        reward = self.config.reward
        sensors = self.config.sensors
        randomization = self.config.domain_randomization
        wind = self.config.wind
        kwargs: dict[str, float | int] = {
            "dt": self.config.environment.dt,
            "action_dim": self.config.action_dim,
            "observation_dim": self.config.observation_dim,
            "observation_flags": self.config.observation_flags,
            "state_noise_std": sensors.state_noise_std,
            "imu_noise_std": sensors.imu_noise_std,
            "task_type": TASK_MAP[task.task_type],
            "action_mode": ACTION_MAP[self.config.environment.action_mode],
            "reset_mode": RESET_MAP[self.config.environment.reset_mode],
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

    def reset(self, seed: int | None = None):
        self._tick = 0
        _binding.vec_reset(self._vec_handle, seed or 0)
        return self.observations, []

    def step(self, actions):
        self._tick += 1
        self.actions[:] = np.asarray(actions, dtype=np.float32)
        _binding.vec_step(self._vec_handle)
        info: list[dict[str, float]] = []
        if self._emit_logs and self._tick % self._report_interval == 0:
            log = _binding.vec_log(self._vec_handle)
            if log:
                info.append(log)
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def snapshot(self, env_index: int = 0) -> dict[str, float]:
        return _binding.env_get(self._handles[env_index])

    def render(self):
        if self.render_mode is None:
            raise ValueError("render_mode is not enabled for this environment")
        frame = self._snapshot_frame()
        if self._renderer is None:
            fps = float(self.metadata.get("render_fps", 30))
            self._renderer = FlightRenderer(self.config, self.render_mode, fps=fps)
        return self._renderer.render(frame)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        if hasattr(self, "_vec_handle"):
            _binding.vec_close(self._vec_handle)

    def _snapshot_frame(self, env_index: int = 0) -> DroneFrame:
        snapshot = self.snapshot(env_index)
        return DroneFrame(
            x=snapshot["x"],
            z=snapshot["z"],
            vx=snapshot["vx"],
            vz=snapshot["vz"],
            ax=snapshot["ax"],
            az=snapshot["az"],
            pitch=snapshot["pitch"],
            pitch_rate=snapshot["pitch_rate"],
            target_x=snapshot["target_x"],
            target_z=snapshot["target_z"],
            wind_x=snapshot["wind_x"],
            wind_z=snapshot["wind_z"],
            distance=snapshot["distance"],
            reward_total=snapshot["reward_total"],
            motor_thrusts=(
                snapshot["motor_front_left"],
                snapshot["motor_front_right"],
                snapshot["motor_rear_left"],
                snapshot["motor_rear_right"],
            ),
            commands=(
                snapshot["command_0"],
                snapshot["command_1"],
                snapshot["command_2"],
                snapshot["command_3"],
            ),
            action_dim=int(snapshot["action_dim"]),
            active_target=int(snapshot["active_target"]),
            target_count=int(snapshot["target_count"]),
        )
