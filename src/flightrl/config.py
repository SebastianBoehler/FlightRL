from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib


MAX_WAYPOINTS = 8
DEFAULT_FIXED_START = (0.0, 2.0)
DEFAULT_FIXED_TARGET = (0.0, 2.0)
DEFAULT_SPAWN_BOUNDS = (-3.0, 3.0, 1.0, 4.0)
DEFAULT_TARGET_BOUNDS = (-4.0, 4.0, 1.0, 6.0)
DEFAULT_FIXED_WAYPOINTS = [(0.0, 2.0)] * MAX_WAYPOINTS


@dataclass(slots=True)
class EnvironmentConfig:
    dt: float = 0.02
    num_envs: int = 256
    action_mode: str = "stabilized_planar"
    reset_mode: str = "random_uniform"


@dataclass(slots=True)
class DroneConfig:
    mass: float = 1.0
    inertia: float = 0.08
    arm_length: float = 0.25
    drag: float = 0.12
    angular_drag: float = 0.08
    gravity: float = 9.81
    hover_thrust: float = 9.81
    thrust_gain: float = 4.5
    max_total_thrust: float = 18.0
    max_pitch_torque: float = 2.5
    actuator_tau: float = 0.08
    max_velocity: float = 10.0
    max_pitch_rate: float = 6.0
    max_pitch_angle: float = 1.2
    floor_z: float = 0.0
    x_limit: float = 8.0
    z_limit: float = 8.0


@dataclass(slots=True)
class SensorConfig:
    include_position: bool = True
    include_velocity: bool = True
    include_attitude: bool = True
    include_angular_velocity: bool = True
    include_target_vector: bool = True
    include_previous_action: bool = True
    include_health: bool = True
    include_ideal_state: bool = False
    include_noisy_state: bool = False
    include_imu: bool = False
    include_range_sensor: bool = False
    include_vision_sensor: bool = False
    state_noise_std: float = 0.01
    imu_noise_std: float = 0.02


@dataclass(slots=True)
class TaskConfig:
    task_type: str = "hover"
    max_steps: int = 512
    sequence_length: int = 4
    hover_hold_steps: int = 25
    success_radius: float = 0.35
    hover_speed_threshold: float = 0.4
    fixed_start: tuple[float, float] = DEFAULT_FIXED_START
    fixed_target: tuple[float, float] = DEFAULT_FIXED_TARGET
    fixed_waypoints: list[tuple[float, float]] = field(default_factory=lambda: list(DEFAULT_FIXED_WAYPOINTS))
    spawn_bounds: tuple[float, float, float, float] = DEFAULT_SPAWN_BOUNDS
    target_bounds: tuple[float, float, float, float] = DEFAULT_TARGET_BOUNDS


@dataclass(slots=True)
class RewardConfig:
    alive_bonus: float = 0.05
    distance_penalty: float = 0.15
    progress_bonus: float = 1.0
    velocity_penalty: float = 0.05
    angular_rate_penalty: float = 0.03
    control_penalty: float = 0.01
    smoothness_penalty: float = 0.02
    success_bonus: float = 8.0
    crash_penalty: float = 8.0
    out_of_bounds_penalty: float = 4.0


@dataclass(slots=True)
class TrainingConfig:
    device: str = "cpu"
    torch_deterministic: bool = True
    cpu_offload: bool = False
    optimizer: str = "adam"
    precision: str = "float32"
    total_timesteps: int = 131072
    batch_size: int = 8192
    bptt_horizon: int = 32
    minibatch_size: int = 2048
    max_minibatch_size: int = 8192
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 2
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    vf_clip_coef: float = 0.2
    max_grad_norm: float = 0.5
    ent_coef: float = 0.001
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    compile: bool = False
    compile_mode: str = "max-autotune-no-cudagraphs"
    compile_fullgraph: bool = True
    prio_alpha: float = 0.8
    prio_beta0: float = 0.2
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0
    checkpoint_interval: int = 10
    data_dir: str = "artifacts"
    iterations: int = 8
    hidden_size: int = 128


@dataclass(slots=True)
class DomainRandomizationConfig:
    enabled: bool = False
    mass_scale: float = 0.05
    drag_scale: float = 0.1
    thrust_scale: float = 0.08
    actuator_tau_scale: float = 0.2
    sensor_noise_scale: float = 0.5


@dataclass(slots=True)
class LoggingConfig:
    report_interval: int = 16
    trajectory_dir: str = "artifacts/trajectories"


@dataclass(slots=True)
class FlightConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    domain_randomization: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def observation_flags(self) -> int:
        sensors = self.sensors
        flags = 0
        mapping = {
            "include_position": 1 << 0,
            "include_velocity": 1 << 1,
            "include_attitude": 1 << 2,
            "include_angular_velocity": 1 << 3,
            "include_target_vector": 1 << 4,
            "include_previous_action": 1 << 5,
            "include_health": 1 << 6,
            "include_ideal_state": 1 << 7,
            "include_noisy_state": 1 << 8,
            "include_imu": 1 << 9,
            "include_range_sensor": 1 << 10,
            "include_vision_sensor": 1 << 11,
        }
        for attr, bit in mapping.items():
            if getattr(sensors, attr):
                flags |= bit
        return flags

    @property
    def observation_dim(self) -> int:
        dims = 0
        dims += 2 if self.sensors.include_position else 0
        dims += 2 if self.sensors.include_velocity else 0
        dims += 2 if self.sensors.include_attitude else 0
        dims += 1 if self.sensors.include_angular_velocity else 0
        dims += 2 if self.sensors.include_target_vector else 0
        dims += 2 if self.sensors.include_previous_action else 0
        dims += 1 if self.sensors.include_health else 0
        dims += 6 if self.sensors.include_ideal_state else 0
        dims += 6 if self.sensors.include_noisy_state else 0
        dims += 3 if self.sensors.include_imu else 0
        if self.sensors.include_range_sensor or self.sensors.include_vision_sensor:
            raise NotImplementedError("Range and vision sensors are placeholders in the MVP")
        return dims


def _merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value
    return target


def _pair(values: Any, default: tuple[float, float]) -> tuple[float, float]:
    if not values:
        return default
    return (float(values[0]), float(values[1]))


def _bounds(values: Any, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if not values:
        return default
    return tuple(float(v) for v in values[:4])  # type: ignore[return-value]


def _waypoints(values: Any) -> list[tuple[float, float]]:
    points = list(DEFAULT_FIXED_WAYPOINTS)
    if not values:
        return points
    for idx, point in enumerate(values[:MAX_WAYPOINTS]):
        points[idx] = (float(point[0]), float(point[1]))
    return points


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> FlightConfig:
    raw = tomllib.loads(Path(path).read_text())
    if overrides:
        _merge(raw, overrides)

    task_data = raw.get("task", {})
    config = FlightConfig(
        environment=EnvironmentConfig(**raw.get("environment", {})),
        drone=DroneConfig(**raw.get("drone", {})),
        sensors=SensorConfig(**raw.get("sensors", {})),
        task=TaskConfig(
            **{k: v for k, v in task_data.items() if k not in {"fixed_start", "fixed_target", "fixed_waypoints", "spawn_bounds", "target_bounds"}},
            fixed_start=_pair(task_data.get("fixed_start"), DEFAULT_FIXED_START),
            fixed_target=_pair(task_data.get("fixed_target"), DEFAULT_FIXED_TARGET),
            fixed_waypoints=_waypoints(task_data.get("fixed_waypoints")),
            spawn_bounds=_bounds(task_data.get("spawn_bounds"), DEFAULT_SPAWN_BOUNDS),
            target_bounds=_bounds(task_data.get("target_bounds"), DEFAULT_TARGET_BOUNDS),
        ),
        reward=RewardConfig(**raw.get("reward", {})),
        training=TrainingConfig(**raw.get("training", {})),
        domain_randomization=DomainRandomizationConfig(**raw.get("domain_randomization", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
    )
    if config.observation_dim <= 0:
        raise ValueError("Observation config produced an empty observation vector")
    return config
