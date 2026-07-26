from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from flightrl.sixdof import SixDofCrazyflieEnv, teacher_actions
from flightrl.sixdof.action_targets import TARGET_SHAPINGS, shape_action_targets
from flightrl.sixdof.bc_regularization import bc_regularization_loss
from flightrl.sixdof.crash_selection import load_replay_npz
from flightrl.sixdof.disturbance import configure_disturbance
from flightrl.sixdof.puffer_observation import scale_previous_action_observation
from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy, load_puffer_sixdof_policy
from flightrl.sixdof.policies import TEACHER_PROFILES
from flightrl.sixdof.replay_loss import replay_sample_weights, weighted_envelope_loss, weighted_mse_loss
from flightrl.sixdof.transfer_selection import build_transfer_replay, prepare_transfer_selection
from flightrl.sixdof.transfer_test import TransferTestConfig
from flightrl.tracking import add_wandb_args, args_config, init_wandb, log_artifacts, log_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior-clone a Puffer-compatible six-DoF policy from the analytic teacher.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--reset-profile", default="obstacle_close_live")
    parser.add_argument("--sensor-profile", default=None)
    parser.add_argument("--physics-profile", default=None)
    parser.add_argument("--domain-randomization", default=None)
    parser.add_argument("--disturbance-profile", default=None)
    parser.add_argument("--teacher-profile", choices=TEACHER_PROFILES, default="default")
    parser.add_argument("--target-shaping", choices=TARGET_SHAPINGS, default="none")
    parser.add_argument("--target-shaping-strength", type=float, default=1.0)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--collect-steps", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--dagger-iterations", type=int, default=0)
    parser.add_argument("--dagger-steps", type=int, default=256)
    parser.add_argument("--dagger-beta", type=float, default=0.0)
    parser.add_argument("--crash-replay-dataset")
    parser.add_argument("--crash-replay-coef", type=float, default=0.0)
    parser.add_argument("--crash-replay-batch-size", type=int, default=1024)
    parser.add_argument("--crash-replay-envelope-coef", type=float, default=0.0)
    parser.add_argument("--crash-replay-action-abs-limit", type=float, default=0.85)
    parser.add_argument("--transfer-replay-log", action="append", default=[])
    parser.add_argument("--failed-transfer-replay-log", action="append", default=[])
    parser.add_argument("--transfer-replay-coef", type=float, default=0.0)
    parser.add_argument("--transfer-replay-batch-size", type=int, default=1024)
    parser.add_argument("--transfer-replay-envelope-coef", type=float, default=0.0)
    parser.add_argument("--transfer-replay-action-abs-limit", type=float, default=0.85)
    parser.add_argument("--policy-envelope-coef", type=float, default=0.0)
    parser.add_argument("--policy-action-abs-limit", type=float, default=0.85)
    parser.add_argument("--open-space-neutral-coef", type=float, default=0.0)
    parser.add_argument("--open-drift-brake-coef", type=float, default=0.0)
    parser.add_argument("--open-space-clearance-m", type=float, default=0.85)
    parser.add_argument("--neutral-speed-m-s", type=float, default=0.20)
    parser.add_argument("--drift-speed-m-s", type=float, default=0.45)
    parser.add_argument("--target-mode", choices=("current_pose", "fixed_origin"), default="current_pose")
    parser.add_argument("--previous-action-observation-scale", type=float, default=1.0)
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
        physics_profile=args.physics_profile,
        domain_randomization=args.domain_randomization,
    )
    env.teacher_profile = args.teacher_profile
    configure_disturbance(env, args.disturbance_profile)
    observations, targets = collect_teacher_dataset(env, args.collect_steps, args.task, args.previous_action_observation_scale, args.target_shaping, args.target_shaping_strength)
    policy = load_puffer_sixdof_policy(args.init_checkpoint) if args.init_checkpoint else new_policy(observations, targets, args)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    crash_replay = load_replay_npz(args.crash_replay_dataset) if args.crash_replay_coef > 0.0 else None
    transfer_replay = transfer_replay_dataset(args)
    run = init_wandb(args, args_config(args, {"samples": int(observations.shape[0]), "observation_dim": int(observations.shape[1])}))
    history = []
    for iteration in range(args.dagger_iterations + 1):
        history.extend(train(policy, optimizer, observations, targets, args, run, iteration, crash_replay, transfer_replay))
        if iteration < args.dagger_iterations:
            policy_obs, policy_targets = collect_policy_dataset(
                env, policy, args.dagger_steps, args.task, args.dagger_beta, args.previous_action_observation_scale, args.target_shaping, args.target_shaping_strength
            )
            observations = torch.cat([observations, policy_obs], dim=0)
            targets = torch.cat([targets, policy_targets], dim=0)
            log_metrics(run, {"dagger/samples": float(observations.shape[0])}, step=iteration)

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), checkpoint)
    report = {
        "checkpoint": str(checkpoint),
        "task": args.task,
        "init_checkpoint": args.init_checkpoint,
        "reset_profile": args.reset_profile,
        "sensor_profile": args.sensor_profile,
        "physics_profile": args.physics_profile,
        "domain_randomization": args.domain_randomization,
        "disturbance_profile": args.disturbance_profile,
        "teacher_profile": args.teacher_profile,
        "target_shaping": args.target_shaping,
        "target_shaping_strength": args.target_shaping_strength,
        "crash_replay_dataset": args.crash_replay_dataset,
        "crash_replay_coef": args.crash_replay_coef,
        "transfer_replay_logs": args.transfer_replay_log,
        "failed_transfer_replay_logs": args.failed_transfer_replay_log,
        "transfer_replay_coef": args.transfer_replay_coef,
        "policy_envelope_coef": args.policy_envelope_coef,
        "policy_action_abs_limit": args.policy_action_abs_limit,
        "open_space_neutral_coef": args.open_space_neutral_coef,
        "open_drift_brake_coef": args.open_drift_brake_coef,
        "open_space_clearance_m": args.open_space_clearance_m,
        "neutral_speed_m_s": args.neutral_speed_m_s,
        "drift_speed_m_s": args.drift_speed_m_s,
        "target_mode": args.target_mode,
        "transfer_replay_samples": int(len(transfer_replay["observations"])) if transfer_replay else 0,
        "transfer_replay_source_rows": int(transfer_replay.get("source_rows", 0)) if transfer_replay else 0,
        "transfer_replay_excluded_source_rows": int(transfer_replay.get("excluded_source_rows", 0)) if transfer_replay else 0,
        "previous_action_observation_scale": args.previous_action_observation_scale,
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


def new_policy(observations: torch.Tensor, targets: torch.Tensor, args: argparse.Namespace) -> PufferSixDofPolicy:
    return PufferSixDofPolicy(
        PufferPolicyMetadata(
            observation_dim=observations.shape[1],
            hidden_size=args.hidden_size,
            action_dim=targets.shape[1],
            num_layers=args.num_layers,
        )
    )


def collect_teacher_dataset(
    env: SixDofCrazyflieEnv,
    steps: int,
    task: str,
    previous_action_observation_scale: float = 1.0,
    target_shaping: str = "none",
    target_shaping_strength: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs, _ = env.reset()
    observations = []
    targets = []
    for _ in range(steps):
        actions = shape_action_targets(env, teacher_actions(env, task=task), target_shaping, target_shaping_strength)
        observations.append(scale_previous_action_observation(obs, previous_action_observation_scale))
        targets.append(actions.copy())
        obs, _reward, terminals, truncations, _ = env.step(actions)
        done = terminals | truncations
        if np.any(done):
            obs = env.reset_done(done).copy()
    return torch.from_numpy(np.concatenate(observations)).float(), torch.from_numpy(np.concatenate(targets)).float()


def collect_policy_dataset(
    env: SixDofCrazyflieEnv,
    policy,
    steps: int,
    task: str,
    beta: float,
    previous_action_observation_scale: float = 1.0,
    target_shaping: str = "none",
    target_shaping_strength: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs, _ = env.reset()
    observations = []
    targets = []
    beta = float(np.clip(beta, 0.0, 1.0))
    for _ in range(steps):
        policy_obs = scale_previous_action_observation(obs.copy(), previous_action_observation_scale)
        labels = shape_action_targets(env, teacher_actions(env, task=task), target_shaping, target_shaping_strength)
        with torch.no_grad():
            policy_actions = policy(torch.from_numpy(policy_obs).float()).cpu().numpy()
        executed = beta * labels + (1.0 - beta) * policy_actions
        observations.append(policy_obs)
        targets.append(labels.copy())
        obs, _reward, terminals, truncations, _ = env.step(executed)
        done = terminals | truncations
        if np.any(done):
            obs = env.reset_done(done).copy()
    return torch.from_numpy(np.concatenate(observations)).float(), torch.from_numpy(np.concatenate(targets)).float()


def train(policy, optimizer, observations: torch.Tensor, targets: torch.Tensor, args: argparse.Namespace, run, iteration: int, crash_replay, transfer_replay) -> list[dict[str, float]]:
    history = []
    count = observations.shape[0]
    for epoch in range(1, args.epochs + 1):
        indices = torch.randperm(count)
        total = 0.0
        for start in range(0, count, args.minibatch_size):
            batch = indices[start : start + args.minibatch_size]
            pred = policy(observations[batch])
            bc_loss = F.mse_loss(pred, targets[batch])
            crash_loss = replay_loss(
                policy,
                crash_replay,
                coef=args.crash_replay_coef,
                batch_size=args.crash_replay_batch_size,
                envelope_coef=args.crash_replay_envelope_coef,
                action_abs_limit=args.crash_replay_action_abs_limit,
            )
            transfer_loss = replay_loss(
                policy,
                transfer_replay,
                coef=args.transfer_replay_coef,
                batch_size=args.transfer_replay_batch_size,
                envelope_coef=args.transfer_replay_envelope_coef,
                action_abs_limit=args.transfer_replay_action_abs_limit,
            )
            regularization = bc_regularization_loss(
                pred,
                observations[batch],
                envelope_coef=args.policy_envelope_coef,
                action_abs_limit=args.policy_action_abs_limit,
                open_space_neutral_coef=args.open_space_neutral_coef,
                open_drift_brake_coef=args.open_drift_brake_coef,
                open_space_clearance_m=args.open_space_clearance_m,
                neutral_speed_m_s=args.neutral_speed_m_s,
                drift_speed_m_s=args.drift_speed_m_s,
            )
            loss = bc_loss + args.crash_replay_coef * crash_loss + args.transfer_replay_coef * transfer_loss + regularization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total += float(bc_loss.detach()) * len(batch)
        global_epoch = iteration * args.epochs + epoch
        entry = {"dagger_iteration": iteration, "epoch": global_epoch, "loss": total / count, "samples": int(count)}
        history.append(entry)
        log_metrics(run, {"train/loss": entry["loss"], "train/samples": float(count)}, step=global_epoch)
        print(f"iteration={iteration} epoch={global_epoch} loss={entry['loss']:.6f} samples={count}", flush=True)
    return history


def transfer_replay_dataset(args: argparse.Namespace):
    if args.transfer_replay_coef <= 0.0:
        return None
    prepared = prepare_transfer_selection(args.transfer_replay_log)
    prepared.extend(prepare_transfer_selection(args.failed_transfer_replay_log, failed_source=True))
    return build_transfer_replay(
        prepared,
        TransferTestConfig(
            task=args.task,
            target_mode=args.target_mode,
            sensor_profile=args.sensor_profile,
            previous_action_observation_scale=args.previous_action_observation_scale,
        ),
        target_shaping=args.target_shaping,
        target_shaping_strength=args.target_shaping_strength,
    )


def replay_loss(policy, replay, *, coef: float, batch_size: int, envelope_coef: float, action_abs_limit: float) -> torch.Tensor:
    if replay is None or coef <= 0.0 or len(replay["observations"]) == 0:
        return next(policy.parameters()).new_tensor(0.0)
    batch_size = min(batch_size, len(replay["observations"]))
    indices = torch.randperm(len(replay["observations"]))[:batch_size]
    prediction = policy(replay["observations"][indices])
    weights = replay_sample_weights(replay, indices)
    loss = weighted_mse_loss(prediction, replay["target_actions"][indices], weights)
    if envelope_coef > 0.0:
        loss = loss + envelope_coef * weighted_envelope_loss(prediction, action_abs_limit, weights)
    return loss


if __name__ == "__main__":
    main()
