from __future__ import annotations

import argparse
import csv
import json
from math import radians
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from flightrl.hardware.sixdof_live_replay import live_env_from_telemetry, target_from_telemetry
from flightrl.sixdof import SixDofCrazyflieEnv, teacher_actions
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy
from flightrl.tracking import add_wandb_args, args_config, init_wandb, log_artifacts, log_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a Puffer six-DoF checkpoint on live Crazyflie replay rows.")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-log", action="append", required=True)
    parser.add_argument("--val-log", action="append", default=[])
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--sim-steps", type=int, default=0)
    parser.add_argument("--sim-rollout-mode", choices=("teacher", "policy_replay"), default="teacher")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--reset-profile", default="obstacle_close_live")
    parser.add_argument("--sensor-profile", default=None)
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.50])
    parser.add_argument("--target-yaw-deg", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--drop-unsafe-live-rows", action="store_true")
    parser.add_argument("--max-abs-tilt-deg", type=float, default=35.0)
    parser.add_argument("--min-zrange-m", type=float, default=0.18)
    parser.add_argument("--min-state-height-m", type=float, default=0.20)
    parser.add_argument("--max-state-height-m", type=float, default=1.20)
    parser.add_argument("--max-speed-m-s", type=float, default=3.0)
    parser.add_argument("--close-range-m", type=float, default=0.35)
    parser.add_argument("--close-sample-weight", type=float, default=1.0)
    parser.add_argument("--action-l2-coef", type=float, default=0.0)
    parser.add_argument("--action-saturation-coef", type=float, default=0.0)
    parser.add_argument("--action-saturation-threshold", type=float, default=0.75)
    parser.add_argument("--target-thrust-min", type=float, default=-1.0)
    parser.add_argument("--target-thrust-max", type=float, default=1.0)
    parser.add_argument("--target-rate-clip-abs", type=float, default=1.0)
    parser.add_argument("--live-previous-action-mode", choices=("zero", "log_action", "policy_replay"), default="zero")
    parser.add_argument("--seed", type=int, default=919)
    add_wandb_args(parser, default_project="FlightRL")
    args = parser.parse_args()

    start = perf_counter()
    torch.manual_seed(args.seed)
    policy = load_puffer_sixdof_policy(args.init_checkpoint)
    train_obs, train_targets, train_weights, train_live_stats = load_live_dataset(args.train_log, args, replay_policy=policy)
    if args.sim_steps > 0:
        sim_obs, sim_targets, sim_weights = load_sim_dataset(args, policy)
        train_obs = torch.cat([train_obs, sim_obs], dim=0)
        train_targets = torch.cat([train_targets, sim_targets], dim=0)
        train_weights = torch.cat([train_weights, sim_weights], dim=0)
    if args.val_log:
        val_obs, val_targets, val_weights, val_live_stats = load_live_dataset(args.val_log, args, replay_policy=policy)
    else:
        val_obs, val_targets, val_weights, val_live_stats = train_obs, train_targets, train_weights, train_live_stats
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    run = init_wandb(args, args_config(args, {"train_samples": len(train_obs), "val_samples": len(val_obs)}))

    history, best_state, best_epoch = train(policy, optimizer, train_obs, train_targets, train_weights, val_obs, val_targets, args, run)
    policy.load_state_dict(best_state)
    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), output)
    report = {
        "checkpoint": str(output),
        "init_checkpoint": args.init_checkpoint,
        "task": args.task,
        "train_logs": args.train_log,
        "val_logs": args.val_log,
        "train_samples": int(len(train_obs)),
        "val_samples": int(len(val_obs)),
        "train_live_filter": train_live_stats,
        "val_live_filter": val_live_stats,
        "sim_steps": args.sim_steps,
        "reset_profile": args.reset_profile,
        "sensor_profile": args.sensor_profile,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "history": history,
        "best_epoch": best_epoch,
        "target_shape": target_shape_config(args),
        "final_train": action_gap(policy, train_obs, train_targets),
        "final_val": action_gap(policy, val_obs, val_targets),
        "final_val_weighted": action_gap(policy, val_obs, val_targets, val_weights),
        "elapsed_s": perf_counter() - start,
        "safety": "Live-replay fine-tune only; evaluate and shadow before direct hardware control.",
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log_artifacts(run, name=output.stem, paths=[output, report_path], artifact_type="model")
    print(f"checkpoint={output}")
    print(f"report={report_path}")


def load_live_dataset(paths: list[str], args, replay_policy=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if args.live_previous_action_mode == "policy_replay" and replay_policy is None:
        raise SystemExit("--live-previous-action-mode=policy_replay requires a replay policy")
    env = SixDofCrazyflieEnv(num_envs=1, seed=args.seed, task=args.task, sensor_profile=args.sensor_profile)
    target = np.asarray(args.target, dtype=np.float32)
    target_yaw = radians(args.target_yaw_deg)
    observations, targets, weights = [], [], []
    stats = {"accepted": 0, "skipped": 0, "close": 0, "paths": list(paths)}
    for path in paths:
        latest: dict[str, float] = {}
        previous_action = np.zeros(4, dtype=np.float32)
        with Path(path).open() as handle:
            for row in csv.DictReader(handle):
                latest.update({key: parse_float(value) for key, value in row.items() if value != ""})
                telemetry = dict(latest)
                if args.drop_unsafe_live_rows and not live_row_allowed(telemetry, args):
                    stats["skipped"] += 1
                    continue
                live_env_from_telemetry(env, telemetry, target=target_from_telemetry(telemetry, target), target_yaw=target_yaw)
                env.previous_action[0] = previous_action
                observation = env.observation()[0].copy()
                observations.append(observation)
                targets.append(shape_targets(teacher_actions(env, task=args.task), args)[0].copy())
                close = min_horizontal_range_m(telemetry) < args.close_range_m
                weights.append(float(args.close_sample_weight if close else 1.0))
                stats["accepted"] += 1
                stats["close"] += int(close)
                previous_action = next_previous_action(observation, telemetry, args, replay_policy)
    if not observations:
        raise SystemExit("no live replay rows accepted")
    return (
        torch.tensor(np.asarray(observations), dtype=torch.float32),
        torch.tensor(np.asarray(targets), dtype=torch.float32),
        torch.tensor(np.asarray(weights), dtype=torch.float32),
        stats,
    )


def load_sim_dataset(args, replay_policy) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    env = SixDofCrazyflieEnv(
        num_envs=args.num_envs,
        seed=args.seed,
        task=args.task,
        reset_profile=args.reset_profile,
        sensor_profile=args.sensor_profile,
    )
    obs, _ = env.reset()
    observations, targets = [], []
    for _ in range(args.sim_steps):
        labels = teacher_actions(env, task=args.task)
        observations.append(obs.copy())
        targets.append(shape_targets(labels, args).copy())
        actions = labels if args.sim_rollout_mode == "teacher" else policy_actions(replay_policy, obs)
        obs, _reward, terminals, truncations, _ = env.step(actions)
        done = terminals | truncations
        if np.any(done):
            obs = env.reset_done(done).copy()
    obs = torch.tensor(np.concatenate(observations), dtype=torch.float32)
    targets = torch.tensor(np.concatenate(targets), dtype=torch.float32)
    return obs, targets, torch.ones(len(obs), dtype=torch.float32)


def parse_float(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def live_row_allowed(row: dict[str, float], args) -> bool:
    if value(row, "sys.isTumbled") > 0.0 or value(row, "sys.canfly", 1.0) <= 0.0:
        return False
    if abs(value(row, "stabilizer.roll")) > args.max_abs_tilt_deg or abs(value(row, "stabilizer.pitch")) > args.max_abs_tilt_deg:
        return False
    z = value(row, "stateEstimate.z", args.target[2])
    if z < args.min_state_height_m or z > args.max_state_height_m:
        return False
    if live_range_m(row, "range.zrange") < args.min_zrange_m:
        return False
    speed = np.linalg.norm([value(row, "stateEstimate.vx"), value(row, "stateEstimate.vy"), value(row, "stateEstimate.vz")])
    return bool(speed <= args.max_speed_m_s)


def value(row: dict[str, float], key: str, default: float = 0.0) -> float:
    return float(row.get(key, default))


def live_range_m(row: dict[str, float], key: str) -> float:
    raw = value(row, key, 4000.0)
    if raw <= 0.0 or not np.isfinite(raw):
        return 4.0
    return 4.0 if raw >= 32000.0 else raw / 1000.0


def min_horizontal_range_m(row: dict[str, float]) -> float:
    return min(live_range_m(row, key) for key in ("range.front", "range.back", "range.left", "range.right"))


def shape_targets(actions: np.ndarray, args) -> np.ndarray:
    shaped = np.asarray(actions, dtype=np.float32).copy()
    shaped[:, 0] = np.clip(shaped[:, 0], args.target_thrust_min, args.target_thrust_max)
    shaped[:, 1:] = np.clip(shaped[:, 1:], -args.target_rate_clip_abs, args.target_rate_clip_abs)
    return shaped


def next_previous_action(observation: np.ndarray, telemetry: dict[str, float], args, replay_policy) -> np.ndarray:
    if args.live_previous_action_mode == "policy_replay":
        with torch.no_grad():
            return replay_policy(torch.tensor(observation[None, :], dtype=torch.float32)).cpu().numpy()[0].astype(np.float32)
    if args.live_previous_action_mode == "log_action":
        return np.asarray(
            [value(telemetry, key) for key in ("action_thrust", "action_roll_rate", "action_pitch_rate", "action_yaw_rate")],
            dtype=np.float32,
        )
    return np.zeros(4, dtype=np.float32)


def policy_actions(policy, observations: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return policy(torch.tensor(observations, dtype=torch.float32)).cpu().numpy().astype(np.float32)


def target_shape_config(args) -> dict[str, float]:
    return {
        "target_thrust_min": float(args.target_thrust_min),
        "target_thrust_max": float(args.target_thrust_max),
        "target_rate_clip_abs": float(args.target_rate_clip_abs),
        "live_previous_action_mode": args.live_previous_action_mode,
        "sim_rollout_mode": args.sim_rollout_mode,
    }


def train(policy, optimizer, train_obs, train_targets, train_weights, val_obs, val_targets, args, run):
    history = []
    best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    best_epoch = 0
    best_val = action_gap(policy, val_obs, val_targets)["l2_p95"]
    count = len(train_obs)
    for epoch in range(1, args.epochs + 1):
        indices = torch.randperm(count)
        total = 0.0
        for start in range(0, count, args.minibatch_size):
            batch = indices[start : start + args.minibatch_size]
            pred = policy(train_obs[batch])
            per_row_loss = torch.mean((pred - train_targets[batch]) ** 2, dim=1)
            loss = torch.sum(per_row_loss * train_weights[batch]) / torch.sum(train_weights[batch])
            if args.action_l2_coef > 0.0:
                loss = loss + args.action_l2_coef * torch.mean(pred * pred)
            if args.action_saturation_coef > 0.0:
                excess = torch.relu(torch.abs(pred) - args.action_saturation_threshold)
                loss = loss + args.action_saturation_coef * torch.mean(excess * excess)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        train_metrics = action_gap(policy, train_obs, train_targets)
        val_metrics = action_gap(policy, val_obs, val_targets)
        entry = {
            "epoch": epoch,
            "loss": total / count,
            "train_l2_p95": train_metrics["l2_p95"],
            "val_l2_p95": val_metrics["l2_p95"],
        }
        history.append(entry)
        if entry["val_l2_p95"] < best_val:
            best_val = entry["val_l2_p95"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
        log_metrics(run, {f"train/{key}": value for key, value in entry.items()}, step=epoch)
        print(f"epoch={epoch} loss={entry['loss']:.6f} val_l2_p95={entry['val_l2_p95']:.4f}", flush=True)
    return history, best_state, best_epoch


def action_gap(policy, observations: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None) -> dict[str, float]:
    with torch.no_grad():
        pred = policy(observations)
    errors = pred - targets
    l2 = torch.linalg.norm(errors, dim=1)
    metrics = {
        "mse": float(torch.mean(errors * errors)),
        "l2_mean": float(torch.mean(l2)),
        "l2_p95": float(torch.quantile(l2, 0.95)),
        "action_abs_max": float(torch.max(torch.abs(pred))),
    }
    if weights is not None:
        weighted = torch.sum(l2 * weights) / torch.sum(weights)
        metrics["weighted_l2_mean"] = float(weighted)
    return metrics


if __name__ == "__main__":
    main()
