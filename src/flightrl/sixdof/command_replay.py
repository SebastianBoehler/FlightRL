from __future__ import annotations

import csv
import json
from math import atan2, asin, pi
from pathlib import Path
from typing import Mapping

import numpy as np

from .env import SixDofEnv, euler_to_quat, quat_to_yaw
from .geometry import BoxRoom
from .policies import action_from_desired_acc


COMMAND_COLUMNS = ("vx_m_s", "vy_m_s", "vz_m_s", "yawrate_deg_s")


def replay_velocity_commands(
    rows: list[dict[str, str]],
    *,
    room: BoxRoom | None = None,
    normalize_origin: bool = True,
    override_z_m: float | None = None,
    hold_z_m: float | None = None,
    velocity_gain: float = 2.5,
    yawrate_scale: float = 1.0,
    command_frame: str = "body",
    yaw_source: str = "logged",
    vx_sign: float = 1.0,
    vy_sign: float = 1.0,
    max_dt_s: float = 0.08,
) -> tuple[list[dict[str, float]], list[dict[str, str]]]:
    prepared = normalize_rows(rows, override_z_m=override_z_m) if normalize_origin else [dict(row) for row in rows]
    if override_z_m is not None and not normalize_origin:
        prepared = override_height(prepared, override_z_m)
    if not prepared:
        return [], []
    env = SixDofEnv(num_envs=1, room=room, seed=0)
    set_env_from_row(env, prepared[0])
    sim_rows = [row_from_env(env, np.zeros(4, dtype=np.float32), prepared[0], 0)]
    for index, row in enumerate(prepared[:-1], start=1):
        next_row = prepared[index]
        dt = min(max(_float(next_row, "host_time_s") - _float(row, "host_time_s"), 1e-3), max_dt_s)
        env.dt = float(dt)
        action = action_from_command_row(
            env,
            row,
            velocity_gain=velocity_gain,
            hold_z_m=hold_z_m,
            yawrate_scale=yawrate_scale,
            command_frame=command_frame,
            yaw_source=yaw_source,
            vx_sign=vx_sign,
            vy_sign=vy_sign,
        )
        env.step(action)
        sim_rows.append(row_from_env(env, action[0], next_row, index))
    return sim_rows, prepared


def action_from_command_row(
    env: SixDofEnv,
    row: Mapping[str, str],
    *,
    velocity_gain: float = 2.5,
    hold_z_m: float | None = None,
    yawrate_scale: float = 1.0,
    command_frame: str = "body",
    yaw_source: str = "logged",
    vx_sign: float = 1.0,
    vy_sign: float = 1.0,
) -> np.ndarray:
    yaw = command_yaw_rad(env, row, yaw_source)
    vx_body = vx_sign * _float(row, "vx_m_s")
    vy_body = vy_sign * _float(row, "vy_m_s")
    vz_world = _float(row, "vz_m_s") if hold_z_m is None else float(np.clip(1.5 * (hold_z_m - env.position[0, 2]), -0.4, 0.4))
    desired_world = desired_velocity_world(vx_body, vy_body, vz_world, yaw, command_frame)
    desired_acc = velocity_gain * (desired_world[None, :] - env.velocity)
    yaw_rate = np.asarray([yawrate_scale * np.deg2rad(_float(row, "yawrate_deg_s"))], dtype=np.float32)
    return action_from_desired_acc(env, desired_acc, yaw_rate)


def command_yaw_rad(env: SixDofEnv, row: Mapping[str, str], yaw_source: str) -> float:
    if yaw_source == "logged":
        return float(np.deg2rad(_float(row, "stabilizer.yaw")))
    if yaw_source == "sim":
        return float(quat_to_yaw(env.quaternion)[0])
    raise ValueError("yaw_source must be 'logged' or 'sim'")


def desired_velocity_world(vx: float, vy: float, vz: float, yaw: float, command_frame: str) -> np.ndarray:
    if command_frame == "world":
        return np.asarray([vx, vy, vz], dtype=np.float32)
    if command_frame == "body":
        return np.asarray([np.cos(yaw) * vx - np.sin(yaw) * vy, np.sin(yaw) * vx + np.cos(yaw) * vy, vz], dtype=np.float32)
    raise ValueError("command_frame must be 'body' or 'world'")


def set_env_from_row(env: SixDofEnv, row: Mapping[str, str]) -> None:
    env.position[0] = [_float(row, "stateEstimate.x"), _float(row, "stateEstimate.y"), _float(row, "stateEstimate.z")]
    env.velocity[0] = [
        _float(row, "stateEstimate.vx"),
        _float(row, "stateEstimate.vy"),
        _float(row, "stateEstimate.vz"),
    ]
    env.quaternion[0] = euler_to_quat(
        np.asarray([np.deg2rad(_float(row, "stabilizer.roll"))], dtype=np.float32),
        np.asarray([np.deg2rad(_float(row, "stabilizer.pitch"))], dtype=np.float32),
        np.asarray([np.deg2rad(_float(row, "stabilizer.yaw"))], dtype=np.float32),
    )[0]
    env.body_rates[0] = 0.0
    env.target_position[0] = env.position[0]
    env.target_yaw[0] = np.deg2rad(_float(row, "stabilizer.yaw"))
    env.step_count[0] = 0
    env._update_ranges()
    env.observations[:] = env.observation()


def normalize_rows(rows: list[dict[str, str]], *, override_z_m: float | None = None) -> list[dict[str, str]]:
    if not rows:
        return []
    t0 = _float(rows[0], "host_time_s")
    x0 = _float(rows[0], "stateEstimate.x")
    y0 = _float(rows[0], "stateEstimate.y")
    prepared = []
    for row in rows:
        item = dict(row)
        item["host_time_s"] = str(_float(row, "host_time_s") - t0)
        item["stateEstimate.x"] = str(_float(row, "stateEstimate.x") - x0)
        item["stateEstimate.y"] = str(_float(row, "stateEstimate.y") - y0)
        if override_z_m is not None:
            item["raw_stateEstimate.z"] = row.get("stateEstimate.z", "")
            item["stateEstimate.z"] = str(override_z_m)
        prepared.append(item)
    return prepared


def override_height(rows: list[dict[str, str]], override_z_m: float) -> list[dict[str, str]]:
    prepared = []
    for row in rows:
        item = dict(row)
        item["raw_stateEstimate.z"] = row.get("stateEstimate.z", "")
        item["stateEstimate.z"] = str(override_z_m)
        prepared.append(item)
    return prepared


def load_box_room(path: str | None) -> BoxRoom | None:
    if not path:
        return None
    report = json.loads(Path(path).read_text())
    estimate = report.get("room_estimate")
    if not estimate:
        raise ValueError(f"room report has no room_estimate: {path}")
    return BoxRoom(
        x_min=float(estimate["x_min"]),
        x_max=float(estimate["x_max"]),
        y_min=float(estimate["y_min"]),
        y_max=float(estimate["y_max"]),
        z_min=float(estimate["z_min"]),
        z_max=float(estimate["z_max"]),
        max_range_m=float(estimate.get("max_range_m", 4.0)),
    )


def row_from_env(env: SixDofEnv, action: np.ndarray, source: Mapping[str, str], step: int) -> dict[str, float]:
    roll, pitch, yaw = quat_to_euler(env.quaternion[0])
    ranges = env.ranges_m[0] * 1000.0
    return {
        "step": float(step),
        "host_time_s": _float(source, "host_time_s"),
        "stateEstimate.x": float(env.position[0, 0]),
        "stateEstimate.y": float(env.position[0, 1]),
        "stateEstimate.z": float(env.position[0, 2]),
        "stateEstimate.vx": float(env.velocity[0, 0]),
        "stateEstimate.vy": float(env.velocity[0, 1]),
        "stateEstimate.vz": float(env.velocity[0, 2]),
        "stabilizer.roll": roll,
        "stabilizer.pitch": pitch,
        "stabilizer.yaw": yaw,
        "range.front": float(ranges[0]),
        "range.back": float(ranges[1]),
        "range.left": float(ranges[2]),
        "range.right": float(ranges[3]),
        "range.up": float(ranges[4]),
        "range.zrange": float(ranges[5]),
        "vx_m_s": _float(source, "vx_m_s"),
        "vy_m_s": _float(source, "vy_m_s"),
        "vz_m_s": _float(source, "vz_m_s"),
        "yawrate_deg_s": _float(source, "yawrate_deg_s"),
        "action_thrust": float(action[0]),
        "action_roll_rate": float(action[1]),
        "action_pitch_rate": float(action[2]),
        "action_yaw_rate": float(action[3]),
    }


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[Mapping[str, str | float]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["host_time_s"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = [float(value) for value in q]
    roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll * 180.0 / pi, pitch * 180.0 / pi, yaw * 180.0 / pi


def _float(row: Mapping[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
