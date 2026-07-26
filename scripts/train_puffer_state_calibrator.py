from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from flightrl.sixdof.crash_selection import load_replay_npz
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy
from flightrl.sixdof.transfer_selection import build_transfer_replay, prepare_transfer_selection
from flightrl.sixdof.transfer_test import TransferTestConfig
from flightrl.tracking import add_wandb_args, args_config, init_wandb, log_artifacts, log_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill state-dependent action calibration into a Puffer checkpoint.")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--clean-target-checkpoint", required=True)
    parser.add_argument("--crash-target-checkpoint", required=True)
    parser.add_argument("--crash-replay-dataset", required=True)
    parser.add_argument("--clean-log", action="append", required=True, help="LABEL:CSV")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clean-weight", type=float, default=1.0)
    parser.add_argument("--crash-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4242)
    add_wandb_args(parser, default_project="FlightRL")
    args = parser.parse_args()

    start = perf_counter()
    torch.manual_seed(args.seed)
    train_obs, train_targets, train_weights, source_counts = build_dataset(args)
    policy = load_puffer_sixdof_policy(args.init_checkpoint)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    run = init_wandb(args, args_config(args, {"samples": int(len(train_obs)), **source_counts}))
    history = train(policy, optimizer, train_obs, train_targets, train_weights, args, run)
    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), output)
    report = {
        "checkpoint": str(output),
        "init_checkpoint": args.init_checkpoint,
        "clean_target_checkpoint": args.clean_target_checkpoint,
        "crash_target_checkpoint": args.crash_target_checkpoint,
        "crash_replay_dataset": args.crash_replay_dataset,
        "clean_logs": args.clean_log,
        "source_counts": source_counts,
        "history": history,
        "final_gap": action_gap(policy, train_obs, train_targets),
        "elapsed_s": perf_counter() - start,
        "safety": "Offline state-calibration distillation only; passing downstream gates does not approve live hardware deployment.",
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log_artifacts(run, name=output.stem, paths=[output, report_path], artifact_type="model")
    print(f"checkpoint={output}")
    print(f"report={report_path}")


def build_dataset(args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    clean_policy = load_puffer_sixdof_policy(args.clean_target_checkpoint)
    crash_policy = load_puffer_sixdof_policy(args.crash_target_checkpoint)
    transfer = build_transfer_replay(prepare_transfer_selection(args.clean_log), TransferTestConfig(task=args.task))
    if transfer is None:
        raise SystemExit("no clean transfer rows loaded")
    crash = load_replay_npz(args.crash_replay_dataset)
    if crash is None:
        raise SystemExit("no crash replay rows loaded")
    clean_obs = transfer["observations"]
    crash_obs = crash["observations"]
    with torch.no_grad():
        clean_targets = clean_policy(clean_obs)
        crash_targets = crash_policy(crash_obs)
    observations = torch.cat([clean_obs, crash_obs], dim=0)
    targets = torch.cat([clean_targets, crash_targets], dim=0)
    weights = torch.cat(
        [
            torch.full((len(clean_obs),), float(args.clean_weight)),
            torch.full((len(crash_obs),), float(args.crash_weight)),
        ],
        dim=0,
    )
    return observations, targets, weights, {"clean_samples": int(len(clean_obs)), "crash_samples": int(len(crash_obs))}


def train(policy, optimizer, observations: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, args, run) -> list[dict[str, float]]:
    history = []
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(observations))
        total = 0.0
        for start in range(0, len(order), args.minibatch_size):
            idx = order[start : start + args.minibatch_size]
            loss = weighted_action_mse(policy(observations[idx]), targets[idx], weights[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        gap = action_gap(policy, observations, targets)
        entry = {"epoch": epoch, "loss": total / len(observations), **gap}
        history.append(entry)
        log_metrics(run, {f"train/{key}": float(value) for key, value in entry.items()}, step=epoch)
        print(f"epoch={epoch} loss={entry['loss']:.6f} l2_p95={entry['l2_p95']:.4f}", flush=True)
    return history


def weighted_action_mse(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_row = torch.mean((prediction - target).pow(2), dim=1)
    scaled = weights / torch.clamp(torch.mean(weights), min=1e-6)
    return torch.mean(per_row * scaled)


def action_gap(policy, observations: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        prediction = policy(observations)
        l2 = torch.linalg.norm(prediction - targets, dim=1)
    return {
        "l2_mean": float(torch.mean(l2)),
        "l2_p95": float(torch.quantile(l2, 0.95)),
        "action_abs_max": float(torch.max(torch.abs(prediction))),
    }


if __name__ == "__main__":
    main()
