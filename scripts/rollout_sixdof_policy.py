from __future__ import annotations

import argparse
import csv
from math import atan2, asin, pi
from pathlib import Path

import numpy as np
import torch

from flightrl.sixdof import SixDofCrazyflieEnv, SixDofPolicy, teacher_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a 6-DoF simulation checkpoint or teacher into Crazyflie-like CSV")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--native-step", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", default="artifacts/trajectories/sixdof_rollout.csv")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None
    task = args.task or (checkpoint.get("task") if checkpoint else "position_yaw")
    env = SixDofCrazyflieEnv(num_envs=1, seed=args.seed, task=task, use_native_step=args.native_step)
    model = load_model(checkpoint) if checkpoint and not args.teacher else None
    obs, _ = env.reset(seed=args.seed)
    rows = []
    for step in range(args.steps):
        if args.teacher:
            actions = teacher_actions(env, task=task)
        else:
            if model is None:
                raise SystemExit("--checkpoint is required unless --teacher is set")
            with torch.no_grad():
                actions = model(torch.from_numpy(obs).float()).cpu().numpy()
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


def load_model(checkpoint: dict) -> SixDofPolicy:
    model = SixDofPolicy(hidden_size=int(checkpoint.get("hidden_size", 128)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


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
