from __future__ import annotations

import argparse
import csv
from math import radians
from pathlib import Path
from time import time

import numpy as np
import torch

from flightrl.hardware.cflib_bridge import require_cflib, sync_crazyflie_context
from flightrl.hardware.config import load_hardware_config
from flightrl.hardware.telemetry import build_log_configs, with_available_log_variables, with_extra_log_variables
from flightrl.sim2real.hardware_approval import hardware_approval_status
from flightrl.sixdof import SixDofCrazyflieEnv, checkpoint_tasks, load_controller_from_checkpoint, teacher_actions
from flightrl.sixdof.controller import executed_action_for_controller
from flightrl.sixdof.env import euler_to_quat
from flightrl.sixdof.observation import augment_observation
from flightrl.sixdof.tasks import append_task_encoding, parse_task_spec, task_indices_for_name


SIXDOF_SHADOW_LOG_VARIABLES = (
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
    "range.zrange",
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.vz",
    "gyro.x",
    "gyro.y",
    "gyro.z",
    "stabilizer.roll",
    "stabilizer.pitch",
    "stabilizer.yaw",
    "pm.vbat",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor a 6-DoF checkpoint against live Crazyflie telemetry without control.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--hardware-config", default="configs/hardware/crazyflie_2_1_brushless.toml")
    parser.add_argument("--approval-manifest", default="artifacts/replay/sim2real_checkpoint_manifest_current_2026-05-20.json")
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.50], metavar=("X", "Y", "Z"))
    parser.add_argument("--target-yaw-deg", type=float, default=0.0)
    parser.add_argument("--output", default="artifacts/crazyflie_logs/sixdof_shadow_monitor.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    controller = load_controller_from_checkpoint(checkpoint)
    tasks = checkpoint_tasks(checkpoint)
    task = parse_task_spec(args.task)[0]
    task_indices = task_indices_for_name(task, tasks, 1)
    approval = hardware_approval_status(args.checkpoint, args.approval_manifest)
    rows = dry_run_rows(controller, checkpoint, tasks, task, task_indices, args, approval) if args.dry_run else live_rows(controller, checkpoint, tasks, task, task_indices, args, approval)
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print("monitor_only=True controls_drone=False")


def live_rows(controller, checkpoint: dict, tasks: tuple[str, ...], task: str, task_indices: np.ndarray, args, approval: dict) -> list[dict]:
    config = with_extra_log_variables(load_hardware_config(args.hardware_config), SIXDOF_SHADOW_LOG_VARIABLES)
    modules = require_cflib()
    latest: dict[str, float] = {}
    rows = []
    state = ShadowState(checkpoint, tasks, task, task_indices, args)
    deadline = time() + args.duration_s
    with sync_crazyflie_context(config, modules) as scf:
        log_config = with_available_log_variables(scf, config)
        with modules.sync_logger_cls(scf, build_log_configs(modules, log_config)) as logger:
            while time() < deadline:
                _timestamp, values, _conf = next(logger)
                latest.update({key: float(value) for key, value in values.items()})
                rows.append(shadow_row(controller, state, latest, approval))
    return rows


def dry_run_rows(controller, checkpoint: dict, tasks: tuple[str, ...], task: str, task_indices: np.ndarray, args, approval: dict) -> list[dict]:
    state = ShadowState(checkpoint, tasks, task, task_indices, args)
    telemetry = {
        "range.front": 220.0,
        "range.back": 1800.0,
        "range.left": 700.0,
        "range.right": 900.0,
        "range.up": 1800.0,
        "range.zrange": args.target[2] * 1000.0,
        "stateEstimate.z": args.target[2],
        "pm.vbat": 3.8,
    }
    return [shadow_row(controller, state, telemetry, approval)]


class ShadowState:
    def __init__(self, checkpoint: dict, tasks: tuple[str, ...], task: str, task_indices: np.ndarray, args) -> None:
        self.env = SixDofCrazyflieEnv(num_envs=1, seed=0, task=task)
        self.tasks = tasks
        self.task = task
        self.task_indices = task_indices
        self.observation_mode = str(checkpoint.get("observation_mode", "base"))
        self.previous_obs = None
        self.previous_action = np.zeros((1, 4), dtype=np.float32)
        self.target = np.asarray(args.target, dtype=np.float32)
        self.target_yaw = radians(args.target_yaw_deg)


def shadow_row(controller, state: ShadowState, telemetry: dict[str, float], approval: dict) -> dict[str, float | str | bool]:
    update_env_from_telemetry(state.env, telemetry, state.target, state.target_yaw)
    obs = state.env.observation()
    model_obs = append_task_encoding(obs.copy(), state.task_indices, len(state.tasks))
    if state.previous_obs is None:
        state.previous_obs = model_obs.copy()
    policy_obs = augment_observation(model_obs, state.previous_obs, state.previous_action, state.observation_mode)
    teacher = teacher_actions(state.env, task=state.task)
    with torch.no_grad():
        raw = controller.model(torch.from_numpy(policy_obs).float()).cpu().numpy()
    executed = executed_action_for_controller(controller.controller, raw, teacher, controller.residual_scale)
    state.previous_obs = model_obs.copy()
    state.previous_action = executed.astype(np.float32)
    return {
        "host_time_s": time(),
        "monitor_only": True,
        "controls_drone": False,
        "hardware_approved": bool(approval["hardware_approved"]),
        "approval_status": str(approval["approval_status"]),
        **telemetry,
        **action_columns("teacher", teacher[0]),
        **action_columns("raw", raw[0]),
        **action_columns("shadow", executed[0]),
    }


def update_env_from_telemetry(env: SixDofCrazyflieEnv, telemetry: dict[str, float], target: np.ndarray, target_yaw: float) -> None:
    env.position[0] = [value(telemetry, "stateEstimate.x"), value(telemetry, "stateEstimate.y"), value(telemetry, "stateEstimate.z")]
    env.velocity[0] = [value(telemetry, "stateEstimate.vx"), value(telemetry, "stateEstimate.vy"), value(telemetry, "stateEstimate.vz")]
    roll = radians(value(telemetry, "stabilizer.roll"))
    pitch = radians(value(telemetry, "stabilizer.pitch"))
    yaw = radians(value(telemetry, "stabilizer.yaw"))
    env.quaternion[0] = euler_to_quat(np.asarray([roll]), np.asarray([pitch]), np.asarray([yaw]))[0]
    env.body_rates[0] = [radians(value(telemetry, "gyro.x")), radians(value(telemetry, "gyro.y")), radians(value(telemetry, "gyro.z"))]
    env.ranges_m[0] = [
        range_m(telemetry, "range.front"),
        range_m(telemetry, "range.back"),
        range_m(telemetry, "range.left"),
        range_m(telemetry, "range.right"),
        range_m(telemetry, "range.up"),
        range_m(telemetry, "range.zrange"),
    ]
    env.target_position[0] = target
    env.target_yaw[0] = target_yaw


def value(telemetry: dict[str, float], key: str) -> float:
    return float(telemetry.get(key, 0.0))


def range_m(telemetry: dict[str, float], key: str) -> float:
    raw = float(telemetry.get(key, 4000.0))
    return 4.0 if raw >= 32000.0 else raw / 1000.0


def action_columns(prefix: str, action: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_thrust": float(action[0]),
        f"{prefix}_roll_rate": float(action[1]),
        f"{prefix}_pitch_rate": float(action[2]),
        f"{prefix}_yaw_rate": float(action[3]),
    }


def write_rows(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["host_time_s"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
