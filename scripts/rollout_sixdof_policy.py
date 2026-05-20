from __future__ import annotations

import argparse
import csv
import json
from math import atan2, asin, pi
from pathlib import Path

import numpy as np
import torch

from flightrl.sixdof import BoxRoom, SixDofCrazyflieEnv, checkpoint_tasks, load_policy_from_checkpoint, teacher_actions
from flightrl.sixdof.observation import augment_observation
from flightrl.sixdof.tasks import append_task_encoding, parse_task_spec, task_indices_for_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a 6-DoF simulation checkpoint or teacher into Crazyflie-like CSV")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--native-step", action="store_true")
    parser.add_argument("--room-report", default=None, help="room summary JSON from summarize_crazyflie_room.py")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", default="artifacts/trajectories/sixdof_rollout.csv")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None
    tasks = checkpoint_tasks(checkpoint) if checkpoint else parse_task_spec(args.task or "position_yaw")
    task = args.task or tasks[0]
    if "," in task or task == "multitask":
        raise SystemExit("--task must select one concrete task for rollout")
    if args.room_report and args.native_step:
        raise SystemExit("--room-report is only supported without --native-step until native room bounds are configurable")
    room = load_room_report(args.room_report) if args.room_report else None
    env = SixDofCrazyflieEnv(num_envs=1, seed=args.seed, task=task, room=room, use_native_step=args.native_step)
    model = load_policy_from_checkpoint(checkpoint) if checkpoint and not args.teacher else None
    obs, _ = env.reset(seed=args.seed)
    task_indices = task_indices_for_name(task, tasks, env.num_envs)
    observation_mode = checkpoint.get("observation_mode", "base") if checkpoint else "base"
    previous_obs = None
    previous_action = np.zeros((env.num_envs, 4), dtype=np.float32)
    rows = []
    for step in range(args.steps):
        if args.teacher:
            actions = teacher_actions(env, task=task)
        else:
            if model is None:
                raise SystemExit("--checkpoint is required unless --teacher is set")
            model_obs = append_task_encoding(obs, task_indices, len(tasks))
            if previous_obs is None:
                previous_obs = model_obs.copy()
            policy_obs = augment_observation(model_obs, previous_obs, previous_action, observation_mode)
            with torch.no_grad():
                actions = model(torch.from_numpy(policy_obs).float()).cpu().numpy()
            previous_obs = model_obs.copy()
            previous_action = actions.copy()
        obs, rewards, terminals, truncations, _info = env.step(actions)
        rows.append(row_from_env(env, actions[0], float(rewards[0]), step))
        if terminals[0] or truncations[0]:
            break
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


def load_checkpoint(path: str | None) -> dict:
    if path is None:
        return {}
    return torch.load(path, map_location="cpu")


def load_room_report(path: str | None) -> BoxRoom | None:
    if not path:
        return None
    report = json.loads(Path(path).read_text())
    estimate = report.get("room_estimate")
    if not estimate:
        raise SystemExit(f"room report has no room_estimate: {path}")
    return BoxRoom(
        x_min=float(estimate["x_min"]),
        x_max=float(estimate["x_max"]),
        y_min=float(estimate["y_min"]),
        y_max=float(estimate["y_max"]),
        z_min=float(estimate["z_min"]),
        z_max=float(estimate["z_max"]),
        max_range_m=float(estimate.get("max_range_m", 4.0)),
    )


def row_from_env(env: SixDofCrazyflieEnv, action: np.ndarray, reward: float, step: int) -> dict[str, float]:
    roll, pitch, yaw = quat_to_euler(env.quaternion[0])
    ranges = env.ranges_m[0] * 1000.0
    return {
        "step": float(step),
        "host_time_s": float(step * env.dt),
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
        "reward": reward,
        "action_thrust": float(action[0]),
        "action_roll_rate": float(action[1]),
        "action_pitch_rate": float(action[2]),
        "action_yaw_rate": float(action[3]),
    }


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in q]
    roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (roll * 180.0 / pi, pitch * 180.0 / pi, yaw * 180.0 / pi)


def write_rows(path: str | Path, rows: list[dict[str, float]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["step"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
