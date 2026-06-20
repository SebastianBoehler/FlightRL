from __future__ import annotations

import argparse
import csv
import json
from math import radians
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from flightrl.hardware.sixdof_live_replay import live_env_from_telemetry
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
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--reset-profile", default="obstacle_close_live")
    parser.add_argument("--sensor-profile", default=None)
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.50])
    parser.add_argument("--target-yaw-deg", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=919)
    add_wandb_args(parser, default_project="FlightRL")
    args = parser.parse_args()

    start = perf_counter()
    torch.manual_seed(args.seed)
    policy = load_puffer_sixdof_policy(args.init_checkpoint)
    train_obs, train_targets = load_live_dataset(args.train_log, args)
    if args.sim_steps > 0:
        sim_obs, sim_targets = load_sim_dataset(args)
        train_obs = torch.cat([train_obs, sim_obs], dim=0)
        train_targets = torch.cat([train_targets, sim_targets], dim=0)
    val_obs, val_targets = load_live_dataset(args.val_log, args) if args.val_log else (train_obs, train_targets)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    run = init_wandb(args, args_config(args, {"train_samples": len(train_obs), "val_samples": len(val_obs)}))

    history = train(policy, optimizer, train_obs, train_targets, val_obs, val_targets, args, run)
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
        "sim_steps": args.sim_steps,
        "reset_profile": args.reset_profile,
        "sensor_profile": args.sensor_profile,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "history": history,
        "final_train": action_gap(policy, train_obs, train_targets),
        "final_val": action_gap(policy, val_obs, val_targets),
        "elapsed_s": perf_counter() - start,
        "safety": "Live-replay fine-tune only; evaluate and shadow before direct hardware control.",
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log_artifacts(run, name=output.stem, paths=[output, report_path], artifact_type="model")
    print(f"checkpoint={output}")
    print(f"report={report_path}")


def load_live_dataset(paths: list[str], args) -> tuple[torch.Tensor, torch.Tensor]:
    env = SixDofCrazyflieEnv(num_envs=1, seed=args.seed, task=args.task)
    target = np.asarray(args.target, dtype=np.float32)
    target_yaw = radians(args.target_yaw_deg)
    observations, targets = [], []
    for path in paths:
        with Path(path).open() as handle:
            for row in csv.DictReader(handle):
                telemetry = {key: parse_float(value) for key, value in row.items()}
                live_env_from_telemetry(env, telemetry, target=target, target_yaw=target_yaw)
                observations.append(env.observation()[0].copy())
                targets.append(teacher_actions(env, task=args.task)[0].copy())
    return torch.tensor(np.asarray(observations), dtype=torch.float32), torch.tensor(np.asarray(targets), dtype=torch.float32)


def load_sim_dataset(args) -> tuple[torch.Tensor, torch.Tensor]:
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
        actions = teacher_actions(env, task=args.task)
        observations.append(obs.copy())
        targets.append(actions.copy())
        obs, _reward, terminals, truncations, _ = env.step(actions)
        done = terminals | truncations
        if np.any(done):
            obs = env.reset_done(done).copy()
    return torch.tensor(np.concatenate(observations), dtype=torch.float32), torch.tensor(np.concatenate(targets), dtype=torch.float32)


def parse_float(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def train(policy, optimizer, train_obs, train_targets, val_obs, val_targets, args, run) -> list[dict[str, float]]:
    history = []
    count = len(train_obs)
    for epoch in range(1, args.epochs + 1):
        indices = torch.randperm(count)
        total = 0.0
        for start in range(0, count, args.minibatch_size):
            batch = indices[start : start + args.minibatch_size]
            pred = policy(train_obs[batch])
            loss = F.mse_loss(pred, train_targets[batch])
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
        log_metrics(run, {f"train/{key}": value for key, value in entry.items()}, step=epoch)
        print(f"epoch={epoch} loss={entry['loss']:.6f} val_l2_p95={entry['val_l2_p95']:.4f}", flush=True)
    return history


def action_gap(policy, observations: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        pred = policy(observations)
    errors = pred - targets
    l2 = torch.linalg.norm(errors, dim=1)
    return {
        "mse": float(torch.mean(errors * errors)),
        "l2_mean": float(torch.mean(l2)),
        "l2_p95": float(torch.quantile(l2, 0.95)),
        "action_abs_max": float(torch.max(torch.abs(pred))),
    }


if __name__ == "__main__":
    main()
