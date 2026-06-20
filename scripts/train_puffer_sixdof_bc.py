from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from flightrl.sixdof import SixDofCrazyflieEnv, teacher_actions
from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy
from flightrl.tracking import add_wandb_args, args_config, init_wandb, log_artifacts, log_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior-clone a Puffer-compatible six-DoF policy from the analytic teacher.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--reset-profile", default="obstacle_close_live")
    parser.add_argument("--sensor-profile", default=None)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--collect-steps", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--dagger-iterations", type=int, default=0)
    parser.add_argument("--dagger-steps", type=int, default=256)
    parser.add_argument("--dagger-beta", type=float, default=0.0)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=818)
    add_wandb_args(parser, default_project="FlightRL")
    args = parser.parse_args()

    start = perf_counter()
    torch.manual_seed(args.seed)
    env = SixDofCrazyflieEnv(
        num_envs=args.num_envs,
        seed=args.seed,
        task=args.task,
        reset_profile=args.reset_profile,
        sensor_profile=args.sensor_profile,
    )
    observations, targets = collect_teacher_dataset(env, args.collect_steps, args.task)
    policy = PufferSixDofPolicy(
        PufferPolicyMetadata(
            observation_dim=observations.shape[1],
            hidden_size=args.hidden_size,
            action_dim=targets.shape[1],
            num_layers=args.num_layers,
        )
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    run = init_wandb(args, args_config(args, {"samples": int(observations.shape[0]), "observation_dim": int(observations.shape[1])}))
    history = []
    for iteration in range(args.dagger_iterations + 1):
        history.extend(train(policy, optimizer, observations, targets, args, run, iteration))
        if iteration < args.dagger_iterations:
            policy_obs, policy_targets = collect_policy_dataset(env, policy, args.dagger_steps, args.task, args.dagger_beta)
            observations = torch.cat([observations, policy_obs], dim=0)
            targets = torch.cat([targets, policy_targets], dim=0)
            log_metrics(run, {"dagger/samples": float(observations.shape[0])}, step=iteration)

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), checkpoint)
    report = {
        "checkpoint": str(checkpoint),
        "task": args.task,
        "reset_profile": args.reset_profile,
        "sensor_profile": args.sensor_profile,
        "samples": int(observations.shape[0]),
        "history": history,
        "final_loss": history[-1]["loss"],
        "elapsed_s": perf_counter() - start,
        "note": "Puffer-compatible behavioral cloning checkpoint; evaluate before any live use.",
    }
    report_path = checkpoint.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log_artifacts(run, name=checkpoint.stem, paths=[checkpoint, report_path], artifact_type="model")
    print(f"checkpoint={checkpoint}")
    print(f"report={report_path}")


def collect_teacher_dataset(env: SixDofCrazyflieEnv, steps: int, task: str) -> tuple[torch.Tensor, torch.Tensor]:
    obs, _ = env.reset()
    observations = []
    targets = []
    for _ in range(steps):
        actions = teacher_actions(env, task=task)
        observations.append(obs.copy())
        targets.append(actions.copy())
        obs, _reward, terminals, truncations, _ = env.step(actions)
        done = terminals | truncations
        if np.any(done):
            obs = env.reset_done(done).copy()
    return torch.from_numpy(np.concatenate(observations)).float(), torch.from_numpy(np.concatenate(targets)).float()


def collect_policy_dataset(env: SixDofCrazyflieEnv, policy, steps: int, task: str, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    obs, _ = env.reset()
    observations = []
    targets = []
    beta = float(np.clip(beta, 0.0, 1.0))
    for _ in range(steps):
        labels = teacher_actions(env, task=task)
        with torch.no_grad():
            policy_actions = policy(torch.from_numpy(obs.copy()).float()).cpu().numpy()
        executed = beta * labels + (1.0 - beta) * policy_actions
        observations.append(obs.copy())
        targets.append(labels.copy())
        obs, _reward, terminals, truncations, _ = env.step(executed)
        done = terminals | truncations
        if np.any(done):
            obs = env.reset_done(done).copy()
    return torch.from_numpy(np.concatenate(observations)).float(), torch.from_numpy(np.concatenate(targets)).float()


def train(policy, optimizer, observations: torch.Tensor, targets: torch.Tensor, args: argparse.Namespace, run, iteration: int) -> list[dict[str, float]]:
    history = []
    count = observations.shape[0]
    for epoch in range(1, args.epochs + 1):
        indices = torch.randperm(count)
        total = 0.0
        for start in range(0, count, args.minibatch_size):
            batch = indices[start : start + args.minibatch_size]
            pred = policy(observations[batch])
            loss = F.mse_loss(pred, targets[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        global_epoch = iteration * args.epochs + epoch
        entry = {"dagger_iteration": iteration, "epoch": global_epoch, "loss": total / count, "samples": int(count)}
        history.append(entry)
        log_metrics(run, {"train/loss": entry["loss"], "train/samples": float(count)}, step=global_epoch)
        print(f"iteration={iteration} epoch={global_epoch} loss={entry['loss']:.6f} samples={count}", flush=True)
    return history


if __name__ == "__main__":
    main()
