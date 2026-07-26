from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from flightrl.sixdof import SixDofCrazyflieEnv, teacher_actions
from flightrl.sixdof.crash_selection import load_replay_npz
from flightrl.sixdof.disturbance import configure_disturbance
from flightrl.sixdof.mode_conditioned import BASE_OBSERVATION_DIM, MODES, append_mode_torch, expand_policy_for_modes, mode_index
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy
from flightrl.sixdof.replay_loss import weighted_envelope_loss, weighted_mse_loss
from flightrl.sixdof.transfer_selection import build_transfer_replay, prepare_transfer_selection
from flightrl.sixdof.transfer_test import TransferTestConfig, load_live_rows
from flightrl.sixdof.velocity_transfer import VelocityTransferConfig, logged_action_vector, update_env_from_velocity_row
from flightrl.tracking import add_wandb_args, args_config, init_wandb, log_artifacts, log_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a two-mode Puffer six-DoF BC policy from live transfer replay.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--reset-profile", default="obstacle_hover_drift_recovery")
    parser.add_argument("--disturbance-profile", default="raw_live_drift")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--collect-steps", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--transfer-replay-log", action="append", default=[])
    parser.add_argument("--failed-transfer-replay-log", action="append", default=[])
    parser.add_argument("--crash-replay-dataset")
    parser.add_argument("--velocity-replay-log", action="append", default=[])
    parser.add_argument("--sim-weight", type=float, default=1.0)
    parser.add_argument("--transfer-weight", type=float, default=3.0)
    parser.add_argument("--crash-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=1.5)
    parser.add_argument("--envelope-coef", type=float, default=0.08)
    parser.add_argument("--action-abs-limit", type=float, default=0.90)
    parser.add_argument("--train-velocity-mode-column-only", action="store_true")
    parser.add_argument("--seed", type=int, default=929)
    add_wandb_args(parser, default_project="FlightRL")
    args = parser.parse_args()

    start = perf_counter()
    torch.manual_seed(args.seed)
    base = load_puffer_sixdof_policy(args.init_checkpoint)
    policy = expand_policy_for_modes(base)
    if args.train_velocity_mode_column_only:
        restrict_to_velocity_mode_column(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    replay = build_dataset(args)
    run = init_wandb(args, args_config(args, {"samples": int(len(replay["observations"])), "observation_dim": policy.metadata.observation_dim, "modes": list(MODES)}))
    history = train(policy, optimizer, replay, args, run)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), checkpoint)
    report = {
        "checkpoint": str(checkpoint),
        "init_checkpoint": args.init_checkpoint,
        "modes": list(MODES),
        "samples": int(len(replay["observations"])),
        "mode_counts": replay["mode_counts"],
        "observation_dim": policy.metadata.observation_dim,
        "train_velocity_mode_column_only": args.train_velocity_mode_column_only,
        "history": history,
        "final_loss": history[-1]["loss"],
        "elapsed_s": perf_counter() - start,
        "note": "Mode-conditioned offline BC checkpoint; requires explicit mode wrapping before evaluation or live use.",
    }
    report_path = checkpoint.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log_artifacts(run, name=checkpoint.stem, paths=[checkpoint, report_path], artifact_type="model")
    print(f"checkpoint={checkpoint}")
    print(f"report={report_path}")


def build_dataset(args: argparse.Namespace) -> dict:
    chunks = [sim_chunk(args)]
    chunks.extend(filter(None, [transfer_chunk(args), crash_chunk(args), velocity_chunk(args)]))
    observations = torch.cat([chunk["observations"] for chunk in chunks], dim=0)
    targets = torch.cat([chunk["targets"] for chunk in chunks], dim=0)
    weights = torch.cat([chunk["weights"] for chunk in chunks], dim=0)
    mode_counts = {mode: int(sum(chunk["mode_counts"].get(mode, 0) for chunk in chunks)) for mode in MODES}
    weights = weights / torch.clamp(weights.mean(), min=1e-6)
    return {"observations": observations, "targets": targets, "weights": weights, "mode_counts": mode_counts}


def sim_chunk(args: argparse.Namespace) -> dict:
    env = SixDofCrazyflieEnv(num_envs=args.num_envs, seed=args.seed, task="obstacle_avoidance", reset_profile=args.reset_profile)
    configure_disturbance(env, args.disturbance_profile)
    obs, _ = env.reset()
    observations, targets = [], []
    for _ in range(args.collect_steps):
        actions = teacher_actions(env, task="obstacle_avoidance")
        observations.append(obs.copy())
        targets.append(actions.copy())
        obs, _reward, terminals, truncations, _ = env.step(actions)
        done = terminals | truncations
        if np.any(done):
            obs = env.reset_done(done).copy()
    return tensor_chunk(np.concatenate(observations), np.concatenate(targets), "obstacle_hover", args.sim_weight)


def transfer_chunk(args: argparse.Namespace) -> dict | None:
    if not args.transfer_replay_log and not args.failed_transfer_replay_log:
        return None
    prepared = prepare_transfer_selection(args.transfer_replay_log)
    prepared.extend(prepare_transfer_selection(args.failed_transfer_replay_log, failed_source=True))
    replay = build_transfer_replay(prepared, TransferTestConfig(task="obstacle_avoidance"))
    if replay is None:
        return None
    weights = replay.get("sample_weights", torch.ones(len(replay["observations"]))) * args.transfer_weight
    return tensor_chunk(replay["observations"], replay["target_actions"], "obstacle_hover", weights)


def crash_chunk(args: argparse.Namespace) -> dict | None:
    replay = load_replay_npz(args.crash_replay_dataset)
    if replay is None:
        return None
    weights = replay.get("sample_weights", torch.ones(len(replay["observations"]))) * args.crash_weight
    return tensor_chunk(replay["observations"], replay["target_actions"], "obstacle_hover", weights)


def velocity_chunk(args: argparse.Namespace) -> dict | None:
    rows = [row for spec in args.velocity_replay_log for row in load_live_rows(spec.split(":", 1)[-1])]
    rows = [row for row in rows if all(key in row for key in ("vx_m_s", "vy_m_s", "vz_m_s", "yawrate_deg_s"))]
    if not rows:
        return None
    config = VelocityTransferConfig()
    env = SixDofCrazyflieEnv(num_envs=1, seed=0, task=config.task)
    previous = np.zeros(4, dtype=np.float32)
    observations, targets = [], []
    for row in rows:
        update_env_from_velocity_row(env, row, config, previous)
        action = logged_action_vector(row)
        observations.append(env.observation()[0].copy())
        targets.append(action.copy())
        previous = action
    return tensor_chunk(np.asarray(observations), np.asarray(targets), "velocity_target", args.velocity_weight)


def tensor_chunk(observations, targets, mode: str, weight) -> dict:
    obs = append_mode_torch(torch.as_tensor(observations, dtype=torch.float32), mode)
    target = torch.as_tensor(targets, dtype=torch.float32)
    weights = torch.as_tensor(weight, dtype=torch.float32)
    if weights.ndim == 0:
        weights = torch.full((len(obs),), float(weights), dtype=torch.float32)
    return {"observations": obs, "targets": target, "weights": weights, "mode_counts": {mode: len(obs)}}


def train(policy, optimizer, replay: dict, args: argparse.Namespace, run) -> list[dict[str, float]]:
    history = []
    count = len(replay["observations"])
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(count)
        total = 0.0
        for start in range(0, count, args.minibatch_size):
            idx = order[start : start + args.minibatch_size]
            prediction = policy(replay["observations"][idx])
            weights = replay["weights"][idx]
            loss = weighted_mse_loss(prediction, replay["targets"][idx], weights)
            if args.envelope_coef > 0.0:
                loss = loss + args.envelope_coef * weighted_envelope_loss(prediction, args.action_abs_limit, weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        entry = {"epoch": epoch, "loss": total / count, "samples": int(count)}
        history.append(entry)
        log_metrics(run, {"train/loss": entry["loss"], "train/samples": float(count)}, step=epoch)
        print(f"epoch={epoch} loss={entry['loss']:.6f} samples={count}", flush=True)
    return history


def restrict_to_velocity_mode_column(policy) -> None:
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    weight = policy.encoder.encoder.weight
    weight.requires_grad_(True)
    velocity_col = BASE_OBSERVATION_DIM + mode_index("velocity_target")

    def mask_gradient(gradient: torch.Tensor) -> torch.Tensor:
        masked = torch.zeros_like(gradient)
        masked[:, velocity_col] = gradient[:, velocity_col]
        return masked

    weight.register_hook(mask_gradient)


if __name__ == "__main__":
    main()
