from __future__ import annotations

import csv
from dataclasses import dataclass
from math import radians
from pathlib import Path
from time import time

import numpy as np
import torch

from flightrl.hardware.avoidance_live import AVOIDANCE_LOG_VARIABLES
from flightrl.hardware.sixdof_live_replay import live_env_from_telemetry, target_from_telemetry
from flightrl.hardware.sixdof_raw_action import RawPufferActionConfig, raw_action_to_manual_setpoint
from flightrl.sixdof.puffer_observation import scale_previous_action_observation
from flightrl.sixdof import SixDofCrazyflieEnv


PUFFER_SHADOW_LOG_VARIABLES = AVOIDANCE_LOG_VARIABLES + (
    "supervisor.info",
    "sys.canfly",
    "sys.isFlying",
    "sys.isTumbled",
    "pm.batteryLevel",
)


@dataclass(frozen=True, slots=True)
class PufferShadowConfig:
    height_m: float = 0.50
    target_yaw_deg: float = 0.0
    previous_action_observation_scale: float = 0.25
    raw_action: RawPufferActionConfig = RawPufferActionConfig()


def puffer_shadow_row(
    policy,
    telemetry: dict[str, float],
    config: PufferShadowConfig,
    *,
    previous_action: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> dict[str, float | bool]:
    env = SixDofCrazyflieEnv(num_envs=1, seed=0, task="obstacle_avoidance")
    fallback_target = np.asarray(target if target is not None else [0.0, 0.0, config.height_m], dtype=np.float32)
    live_target = target_from_telemetry(telemetry, fallback_target)
    live_env_from_telemetry(env, telemetry, target=live_target, target_yaw=radians(config.target_yaw_deg))
    if previous_action is not None:
        env.previous_action[0] = np.asarray(previous_action, dtype=np.float32)
    with torch.no_grad():
        observation = scale_previous_action_observation(env.observation(), config.previous_action_observation_scale)
        action = policy(torch.from_numpy(observation).float()).cpu().numpy()[0]
    setpoint = raw_action_to_manual_setpoint(action, config.raw_action)
    return {
        **telemetry,
        "host_time_s": float(telemetry.get("host_time_s", time()) or time()),
        "monitor_only": True,
        "controls_drone": False,
        "raw_puffer_output": True,
        "target_x": float(live_target[0]),
        "target_y": float(live_target[1]),
        "target_z": float(live_target[2]),
        "action_thrust": float(action[0]),
        "action_roll_rate": float(action[1]),
        "action_pitch_rate": float(action[2]),
        "action_yaw_rate": float(action[3]),
        "roll_rate_deg_s": setpoint.roll_rate_deg_s,
        "pitch_rate_deg_s": setpoint.pitch_rate_deg_s,
        "commander_pitch_rate_deg_s": setpoint.commander_pitch_rate_deg_s,
        "yaw_rate_deg_s": setpoint.yaw_rate_deg_s,
        "thrust_percent": setpoint.thrust_percent,
    }


def synthetic_puffer_shadow_telemetry(config: PufferShadowConfig) -> dict[str, float]:
    return {
        "range.front": 260.0,
        "range.back": 1800.0,
        "range.left": 900.0,
        "range.right": 900.0,
        "range.up": 1500.0,
        "range.zrange": config.height_m * 1000.0,
        "stateEstimate.z": config.height_m,
        "pm.vbat": 3.85,
        "host_time_s": time(),
    }


def write_rows(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["host_time_s"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
