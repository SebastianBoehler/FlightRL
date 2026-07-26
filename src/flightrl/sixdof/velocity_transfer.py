from __future__ import annotations

from dataclasses import asdict, dataclass
from math import radians
from typing import Any

import numpy as np
import torch

from flightrl.hardware.sixdof_live_replay import range_m, value
from flightrl.hardware.sixdof_velocity_adapter import SixDofVelocityAdapterConfig, sixdof_action_to_velocity_command
from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.env import euler_to_quat


ACTION_COLUMNS = ("action_thrust", "action_roll_rate", "action_pitch_rate", "action_yaw_rate")
COMMAND_COLUMNS = ("vx_m_s", "vy_m_s", "vz_m_s", "yawrate_deg_s")


@dataclass(frozen=True, slots=True)
class VelocityTransferConfig:
    task: str = "obstacle_avoidance"
    target_height_m: float = 0.50
    min_samples: int = 100
    max_horizontal_l2_p95_m_s: float = 0.08
    max_velocity_l2_p95_m_s: float = 0.09
    max_yaw_abs_p95_deg_s: float = 6.0
    min_vx_sign_agreement: float = 0.55
    min_vy_sign_agreement: float = 0.55
    min_yaw_sign_agreement: float = 0.35
    sign_min_abs_m_s: float = 0.005
    yaw_sign_min_abs_deg_s: float = 0.5
    max_horizontal_speed_m_s: float = 0.12
    max_vertical_speed_m_s: float = 0.04
    max_yawrate_deg_s: float = 12.0
    rate_horizon_s: float = 0.08
    max_virtual_tilt_rad: float = 0.18
    horizontal_gain_s: float = 0.06
    policy_blend: float = 1.0


def score_velocity_transfer_policy(policy, rows: list[dict[str, float]], config: VelocityTransferConfig) -> dict[str, Any]:
    scored = command_rows(rows)
    policy_commands, source_commands, target_commands = replay_commands(policy, scored, config)
    policy_metrics = command_metrics(policy_commands, target_commands, config)
    source_metrics = command_metrics(source_commands, target_commands, config) if len(source_commands) else {"samples": 0}
    return {
        "samples": len(scored),
        "policy": policy_metrics,
        "source_adapter": source_metrics,
        "gate": velocity_gate(policy_metrics, source_metrics, config),
        "config": asdict(config),
    }


def replay_commands(policy, rows: list[dict[str, float]], config: VelocityTransferConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = SixDofCrazyflieEnv(num_envs=1, seed=0, task=config.task)
    adapter = adapter_config(config)
    policy_previous = np.zeros(4, dtype=np.float32)
    source_previous = np.zeros(4, dtype=np.float32)
    policy_commands = []
    source_commands = []
    targets = []
    with torch.no_grad():
        for row in rows:
            update_env_from_velocity_row(env, row, config, policy_previous)
            action = policy(torch.from_numpy(env.observation()).float()).cpu().numpy()[0].astype(np.float32)
            command = sixdof_action_to_velocity_command(env, action, adapter)
            policy_commands.append(command_vector(command))
            if all(column in row for column in ACTION_COLUMNS):
                update_env_from_velocity_row(env, row, config, source_previous)
                logged_action = logged_action_vector(row)
                source = sixdof_action_to_velocity_command(env, logged_action, adapter)
                source_commands.append(command_vector(source))
                source_previous = logged_action
            targets.append([value(row, column) for column in COMMAND_COLUMNS])
            policy_previous = action
    return (
        np.asarray(policy_commands, dtype=np.float32),
        np.asarray(source_commands, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def update_env_from_velocity_row(
    env: SixDofCrazyflieEnv,
    row: dict[str, float],
    config: VelocityTransferConfig,
    previous_action: np.ndarray,
) -> None:
    target_z = value(row, "target_z") or config.target_height_m
    env.position[0] = [
        value(row, "stateEstimate.x"),
        value(row, "stateEstimate.y"),
        value(row, "stateEstimate.z") or target_z,
    ]
    env.velocity[0] = [
        value(row, "stateEstimate.vx"),
        value(row, "stateEstimate.vy"),
        value(row, "stateEstimate.vz"),
    ]
    env.quaternion[0] = euler_to_quat(
        np.asarray([radians(value(row, "stabilizer.roll"))]),
        np.asarray([radians(value(row, "stabilizer.pitch"))]),
        np.asarray([radians(value(row, "stabilizer.yaw"))]),
    )[0]
    env.body_rates[0] = [
        radians(value(row, "gyro.x")),
        radians(value(row, "gyro.y")),
        radians(value(row, "gyro.z")),
    ]
    env.ranges_m[0] = [range_m(row, key) for key in ("range.front", "range.back", "range.left", "range.right", "range.up", "range.zrange")]
    env.target_position[0] = [value(row, "target_x"), value(row, "target_y"), target_z]
    env.target_yaw[0] = radians(value(row, "stabilizer.yaw"))
    env.previous_action[0] = previous_action


def command_rows(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    return [row for row in rows if all(column in row for column in COMMAND_COLUMNS)]


def command_metrics(commands: np.ndarray, targets: np.ndarray, config: VelocityTransferConfig) -> dict[str, Any]:
    if len(commands) == 0:
        return {"samples": 0}
    velocity_error = commands[:, :3] - targets[:, :3]
    horizontal_error = commands[:, :2] - targets[:, :2]
    yaw_error = np.abs(commands[:, 3] - targets[:, 3])
    return {
        "samples": int(len(commands)),
        "velocity_l2_p95_m_s": float(np.quantile(np.linalg.norm(velocity_error, axis=1), 0.95)),
        "horizontal_l2_p95_m_s": float(np.quantile(np.linalg.norm(horizontal_error, axis=1), 0.95)),
        "yaw_abs_p95_deg_s": float(np.quantile(yaw_error, 0.95)),
        "command_abs_max": float(np.max(np.abs(commands))),
        "sign_agreement": {
            "vx": sign_agreement(commands[:, 0], targets[:, 0], config.sign_min_abs_m_s),
            "vy": sign_agreement(commands[:, 1], targets[:, 1], config.sign_min_abs_m_s),
            "vz": sign_agreement(commands[:, 2], targets[:, 2], config.sign_min_abs_m_s),
            "yawrate": sign_agreement(commands[:, 3], targets[:, 3], config.yaw_sign_min_abs_deg_s),
        },
    }


def velocity_gate(policy: dict[str, Any], source: dict[str, Any], config: VelocityTransferConfig) -> dict[str, Any]:
    failures: list[str] = []
    if policy.get("samples", 0) < config.min_samples:
        failures.append("velocity_samples")
    if policy.get("horizontal_l2_p95_m_s", 0.0) > config.max_horizontal_l2_p95_m_s:
        failures.append("velocity_horizontal_l2_p95")
    if policy.get("velocity_l2_p95_m_s", 0.0) > config.max_velocity_l2_p95_m_s:
        failures.append("velocity_l2_p95")
    if policy.get("yaw_abs_p95_deg_s", 0.0) > config.max_yaw_abs_p95_deg_s:
        failures.append("velocity_yaw_abs_p95")
    signs = policy.get("sign_agreement", {})
    if signs.get("vx", 1.0) < config.min_vx_sign_agreement:
        failures.append("velocity_vx_sign")
    if signs.get("vy", 1.0) < config.min_vy_sign_agreement:
        failures.append("velocity_vy_sign")
    if signs.get("yawrate", 1.0) < config.min_yaw_sign_agreement:
        failures.append("velocity_yawrate_sign")
    if source.get("samples", 0) >= config.min_samples and source.get("horizontal_l2_p95_m_s", 0.0) > config.max_horizontal_l2_p95_m_s:
        failures.append("source_adapter_horizontal_l2_p95")
    if source.get("samples", 0) >= config.min_samples and source.get("yaw_abs_p95_deg_s", 0.0) > config.max_yaw_abs_p95_deg_s:
        failures.append("source_adapter_yaw_abs_p95")
    return {"passed": not failures, "failures": failures}


def adapter_config(config: VelocityTransferConfig) -> SixDofVelocityAdapterConfig:
    return SixDofVelocityAdapterConfig(
        max_horizontal_speed_m_s=config.max_horizontal_speed_m_s,
        max_vertical_speed_m_s=config.max_vertical_speed_m_s,
        max_yawrate_deg_s=config.max_yawrate_deg_s,
        rate_horizon_s=config.rate_horizon_s,
        max_virtual_tilt_rad=config.max_virtual_tilt_rad,
        horizontal_gain_s=config.horizontal_gain_s,
        policy_blend=config.policy_blend,
    )


def logged_action_vector(row: dict[str, float]) -> np.ndarray:
    return np.asarray([value(row, column) for column in ACTION_COLUMNS], dtype=np.float32)


def command_vector(command) -> list[float]:
    return [command.vx_m_s, command.vy_m_s, command.vz_m_s, command.yawrate_deg_s]


def sign_agreement(actual: np.ndarray, expected: np.ndarray, min_abs: float) -> float:
    mask = np.abs(expected) > min_abs
    if not np.any(mask):
        return 1.0
    return float(np.mean(np.sign(actual[mask]) == np.sign(expected[mask])))
