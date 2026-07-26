from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.crash_selection import crash_replay_selection_metrics, crash_replay_selection_score, load_replay_npz, replay_samples
from flightrl.sixdof.disturbance_curriculum import add_disturbance_curriculum_args, configure_training_disturbance, disturbance_curriculum_context
from flightrl.sixdof.puffer_evaluation import PufferEvalConfig, evaluate_puffer_backends
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy
from flightrl.sixdof.puffer_ppo import PUFFER_REWARD_MODES, PufferPpoConfig, collect_puffer_rollout, frozen_puffer_policy, puffer_ppo_update
from flightrl.sixdof.policies import TEACHER_PROFILES
from flightrl.sixdof.transfer_selection import (
    build_transfer_replay,
    numeric_metrics,
    prepare_transfer_selection,
    transfer_shadow_selection_metrics,
    transfer_shadow_selection_score,
)
from flightrl.sixdof.transfer_test import TransferTestConfig
from flightrl.tracking import add_wandb_args, args_config, init_wandb, log_artifacts, log_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop PPO fine-tune for Puffer-compatible six-DoF checkpoints.")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--reset-profile", default="obstacle_hover_live")
    parser.add_argument("--eval-reset-profile", action="append", default=None)
    parser.add_argument("--sensor-profile", default=None)
    parser.add_argument("--physics-profile", default=None)
    parser.add_argument("--domain-randomization", default=None)
    parser.add_argument("--disturbance-profile", default=None)
    parser.add_argument("--eval-disturbance-profile", action="append", default=None)
    add_disturbance_curriculum_args(parser)
    parser.add_argument("--teacher-profile", choices=TEACHER_PROFILES, default="default")
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--action-std", type=float, default=0.08)
    parser.add_argument("--imitation-coef", type=float, default=0.0)
    parser.add_argument("--reference-coef", type=float, default=0.2)
    parser.add_argument("--crash-replay-dataset")
    parser.add_argument("--crash-replay-coef", type=float, default=0.0)
    parser.add_argument("--crash-replay-batch-size", type=int, default=512)
    parser.add_argument("--crash-replay-envelope-coef", type=float, default=0.0)
    parser.add_argument("--crash-replay-action-abs-limit", type=float, default=0.85)
    parser.add_argument("--crash-replay-selection-coef", type=float, default=1.0)
    parser.add_argument("--transfer-selection-log", action="append", default=[])
    parser.add_argument("--failed-transfer-selection-log", action="append", default=[])
    parser.add_argument("--transfer-selection-coef", type=float, default=1.0)
    parser.add_argument("--transfer-replay-coef", type=float, default=0.0)
    parser.add_argument("--transfer-replay-batch-size", type=int, default=1024)
    parser.add_argument("--transfer-replay-envelope-coef", type=float, default=0.0)
    parser.add_argument("--transfer-replay-action-abs-limit", type=float, default=0.85)
    parser.add_argument("--previous-action-observation-scale", type=float, default=1.0)
    parser.add_argument("--reward-mode", default="live_stable_clearance", choices=PUFFER_REWARD_MODES)
    parser.add_argument("--selection-backend", choices=("python", "mujoco", "both"), default="python")
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=300)
    parser.add_argument("--eval-num-envs", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=707)
    parser.add_argument("--max-open-space-horizontal-speed-p95-m-s", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=1223)
    parser.add_argument("--native-step", action="store_true")
    add_wandb_args(parser, default_project="FlightRL")
    args = parser.parse_args()

    start = perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy = load_puffer_sixdof_policy(args.init_checkpoint)
    env = SixDofCrazyflieEnv(
        num_envs=args.num_envs,
        seed=args.seed,
        task=args.task,
        use_native_step=args.native_step,
        reset_profile=args.reset_profile,
        sensor_profile=args.sensor_profile,
        physics_profile=args.physics_profile,
        domain_randomization=args.domain_randomization,
    )
    env.teacher_profile = args.teacher_profile
    config = PufferPpoConfig(
        learning_rate=args.learning_rate,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        action_std=args.action_std,
        imitation_coef=args.imitation_coef,
        reference_coef=args.reference_coef,
        crash_replay_coef=args.crash_replay_coef,
        crash_replay_batch_size=args.crash_replay_batch_size,
        crash_replay_envelope_coef=args.crash_replay_envelope_coef,
        crash_replay_action_abs_limit=args.crash_replay_action_abs_limit,
        transfer_replay_coef=args.transfer_replay_coef,
        transfer_replay_batch_size=args.transfer_replay_batch_size,
        transfer_replay_envelope_coef=args.transfer_replay_envelope_coef,
        transfer_replay_action_abs_limit=args.transfer_replay_action_abs_limit,
        previous_action_observation_scale=args.previous_action_observation_scale,
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    reference = frozen_puffer_policy(policy) if args.reference_coef > 0.0 else None
    crash_replay = load_replay_npz(args.crash_replay_dataset) if args.crash_replay_coef > 0.0 else None
    transfer_selection = prepared_transfer_logs(args)
    transfer_config = TransferTestConfig(
        task=args.task,
        sensor_profile=args.sensor_profile,
        previous_action_observation_scale=args.previous_action_observation_scale,
    )
    transfer_replay = build_transfer_replay(transfer_selection, transfer_config) if args.transfer_replay_coef > 0.0 else None
    run = init_wandb(args, args_config(args, {"observation_dim": policy.metadata.observation_dim, "action_dim": policy.metadata.action_dim}))
    interval = args.eval_interval or max(1, args.updates // 4)
    best, entry, metrics = selection_candidate(policy, args, transfer_selection, transfer_config, crash_replay, update=0)
    history = [entry]
    log_metrics(run, wandb_metrics({}, best["reports"], best["selection_score"], best["crash_replay_selection"], best["transfer_shadow_selection"]), step=0)
    print(f"update=0 score={best['selection_score']:.3f} reward={metrics['mean_reward']:.3f} pos_err={metrics['mean_position_error_m']:.3f} open_speed={metrics.get('open_space_horizontal_speed_p95_m_s', 0.0):.3f}", flush=True)
    for update in range(1, args.updates + 1):
        configure_training_disturbance(env, args, update=update, total_updates=args.updates)
        rollout = collect_puffer_rollout(env, policy, horizon=args.horizon, action_std=args.action_std, reward_mode=args.reward_mode, previous_action_observation_scale=args.previous_action_observation_scale)
        losses = puffer_ppo_update(policy, optimizer, rollout, config, reference, crash_replay, transfer_replay)
        if update == 1 or update == args.updates or update % interval == 0:
            candidate, entry, metrics = selection_candidate(policy, args, transfer_selection, transfer_config, crash_replay, update=update, losses=losses)
            history.append(entry)
            log_metrics(run, wandb_metrics(losses, candidate["reports"], candidate["selection_score"], candidate["crash_replay_selection"], candidate["transfer_shadow_selection"]), step=update)
            print(f"update={update} score={candidate['selection_score']:.3f} reward={metrics['mean_reward']:.3f} pos_err={metrics['mean_position_error_m']:.3f} open_speed={metrics.get('open_space_horizontal_speed_p95_m_s', 0.0):.3f}", flush=True)
            if candidate["selection_score"] > best["selection_score"]:
                best = candidate
    write_outputs(best, history, args, perf_counter() - start, run)
def eval_config(args: argparse.Namespace, *, seed: int, disturbance_profile: str | None = None) -> PufferEvalConfig:
    return PufferEvalConfig(
        task=args.task,
        backend=args.selection_backend,
        steps=args.eval_steps,
        num_envs=args.eval_num_envs,
        seed=seed,
        reset_profile=eval_reset_profiles(args)[0],
        sensor_profile=args.sensor_profile,
        physics_profile=args.physics_profile,
        domain_randomization=args.domain_randomization,
        disturbance_profile=disturbance_profile if disturbance_profile is not None else args.disturbance_profile,
        max_open_space_horizontal_speed_p95_m_s=args.max_open_space_horizontal_speed_p95_m_s,
        previous_action_observation_scale=args.previous_action_observation_scale,
    )

def eval_reset_profiles(args: argparse.Namespace) -> list[str]: return args.eval_reset_profile or ["obstacle_hover_live"]
def eval_disturbance_profiles(args: argparse.Namespace) -> list[str | None]: return args.eval_disturbance_profile or [args.disturbance_profile]
def prepared_transfer_logs(args: argparse.Namespace):
    prepared = prepare_transfer_selection(args.transfer_selection_log)
    prepared.extend(prepare_transfer_selection(args.failed_transfer_selection_log, failed_source=True))
    return prepared

def evaluate_selection_reports(policy, args: argparse.Namespace, *, seed: int) -> dict[str, dict]:
    reports = {}
    for profile in eval_reset_profiles(args):
        for disturbance in eval_disturbance_profiles(args):
            config = replace(eval_config(args, seed=seed, disturbance_profile=disturbance), reset_profile=profile)
            for backend, item in evaluate_puffer_backends(policy, config).items():
                item = {**item, "reset_profile": profile, "disturbance_profile": disturbance}
                reports[f"{profile}/{disturbance or 'none'}/{backend}"] = item
    return reports

def selection_candidate(policy, args: argparse.Namespace, transfer_selection, transfer_config: TransferTestConfig, crash_replay, *, update: int, losses: dict | None = None):
    reports = evaluate_selection_reports(policy, args, seed=args.eval_seed)
    crash_metrics = crash_replay_selection_metrics(policy, crash_replay, action_abs_limit=args.crash_replay_action_abs_limit, previous_action_observation_scale=args.previous_action_observation_scale)
    transfer_metrics = transfer_shadow_selection_metrics(policy, transfer_selection, transfer_config)
    score = score_reports(reports) + args.crash_replay_selection_coef * crash_replay_selection_score(crash_metrics, action_abs_limit=args.crash_replay_action_abs_limit) + args.transfer_selection_coef * transfer_shadow_selection_score(transfer_metrics)
    entry = {"update": update, "selection_score": score, **(losses or {}), **summary_report_metrics(reports)}
    entry.update(crash_metrics)
    entry.update(numeric_metrics(transfer_metrics))
    return checkpoint_payload(policy, args, reports, score, update, crash_metrics, transfer_metrics), entry, selection_metrics(reports)

def score_metrics(metrics: dict) -> float:
    return (
        3.0 * metrics["mean_completed_fraction"]
        + metrics["mean_survival_fraction"]
        + metrics["clearance_p01_m"]
        - metrics["mean_position_error_m"]
        - 0.35 * metrics.get("horizontal_speed_p95_m_s", 0.0)
        - 2.0 * metrics.get("open_space_horizontal_speed_p95_m_s", 0.0)
        - 0.04 * metrics.get("tilt_p95_deg", 0.0)
        - metrics.get("action_saturation_fraction", 0.0)
    )
def score_reports(reports: dict[str, dict]) -> float:
    scores = []
    for item in reports.values():
        if item.get("status") != "ok":
            scores.append(-1_000.0)
            continue
        failures = item["gate"]["failures"]
        scores.append(score_metrics(item["metrics"]) - 12.0 * len(failures))
    return float(min(scores))
def selection_metrics(reports: dict[str, dict]) -> dict:
    ok_reports = [item for item in reports.values() if item.get("status") == "ok"]
    if not ok_reports:
        raise ValueError("no successful selection backend reports")
    return min(ok_reports, key=lambda item: score_metrics(item["metrics"]))["metrics"]
def checkpoint_payload(policy, args: argparse.Namespace, reports: dict[str, dict], score: float, update: int, crash_metrics: dict[str, float], transfer_metrics: dict) -> dict:
    return {
        "state_dict": {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()},
        "metrics": selection_metrics(reports),
        "reports": reports,
        "selection_score": float(score),
        "selection_update": int(update),
        "selection_backend": args.selection_backend,
        "task": args.task,
        "reset_profile": args.reset_profile,
        "eval_reset_profiles": eval_reset_profiles(args),
        "sensor_profile": args.sensor_profile,
        "physics_profile": args.physics_profile,
        "domain_randomization": args.domain_randomization,
        "disturbance_profile": args.disturbance_profile,
        "eval_disturbance_profiles": eval_disturbance_profiles(args),
        "disturbance_curriculum": disturbance_curriculum_context(args),
        "teacher_profile": args.teacher_profile,
        "crash_replay_dataset": args.crash_replay_dataset,
        "crash_replay_coef": args.crash_replay_coef,
        "crash_replay_envelope_coef": args.crash_replay_envelope_coef,
        "crash_replay_action_abs_limit": args.crash_replay_action_abs_limit,
        "crash_replay_selection": crash_metrics,
        "transfer_selection_logs": args.transfer_selection_log,
        "failed_transfer_selection_logs": args.failed_transfer_selection_log,
        "transfer_selection_coef": args.transfer_selection_coef,
        "transfer_replay_coef": args.transfer_replay_coef,
        "transfer_replay_envelope_coef": args.transfer_replay_envelope_coef,
        "transfer_replay_action_abs_limit": args.transfer_replay_action_abs_limit,
        "previous_action_observation_scale": args.previous_action_observation_scale,
        "transfer_shadow_selection": transfer_metrics,
        "reward_mode": args.reward_mode,
        "trainer": "puffer_closed_loop_ppo",
        "note": "Puffer-compatible closed-loop PPO checkpoint; not approved for live hardware.",
    }

def write_outputs(best: dict, history: list[dict], args: argparse.Namespace, elapsed_s: float, run) -> None:
    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best["state_dict"], output)
    report = {
        "checkpoint": str(output),
        "init_checkpoint": args.init_checkpoint,
        "selection_update": best["selection_update"],
        "selection_score": best["selection_score"],
        "selection_backend": best["selection_backend"],
        "metrics": best["metrics"],
        "reports": best["reports"],
        "selection_context": {key: best.get(key) for key in ("reward_mode", "teacher_profile", "eval_disturbance_profiles", "disturbance_curriculum", "transfer_selection_logs", "failed_transfer_selection_logs", "transfer_shadow_selection")},
        "gate": combined_gate(best["reports"]),
        "history": history,
        "elapsed_s": elapsed_s,
        "crash_replay_samples": replay_samples(args.crash_replay_dataset),
        "transfer_replay_samples": int(
            len(
                build_transfer_replay(
                    prepared_transfer_logs(args),
                    TransferTestConfig(
                        task=args.task,
                        sensor_profile=args.sensor_profile,
                        previous_action_observation_scale=args.previous_action_observation_scale,
                    ),
                )["observations"]
            )
        )
        if args.transfer_replay_coef > 0.0
        else 0,
        "safety": "Offline closed-loop simulation training only; passing this report does not approve live hardware deployment.",
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log_artifacts(run, name=output.stem, paths=[output, report_path], artifact_type="model")
    print(f"checkpoint={output}")
    print(f"report={report_path}")


def combined_gate(reports: dict[str, dict]) -> dict:
    failures = []
    for backend, item in reports.items():
        if item.get("status") != "ok":
            failures.append(f"{backend}:{item.get('status', 'unknown')}")
            continue
        failures.extend(f"{backend}:{failure}" for failure in item["gate"]["failures"])
    return {"passed": not failures, "failures": failures}


def summary_metrics(metrics: dict) -> dict[str, float]:
    keys = ("mean_reward", "mean_position_error_m", "clearance_p01_m", "mean_completed_fraction", "open_space_horizontal_speed_p95_m_s", "tilt_p95_deg")
    return {key: float(metrics[key]) for key in keys if key in metrics}


def summary_report_metrics(reports: dict[str, dict]) -> dict[str, float]:
    summary = {}
    for backend, item in reports.items():
        if item.get("status") != "ok":
            continue
        for key, value in summary_metrics(item["metrics"]).items():
            summary[f"{backend}_{key}"] = value
        summary[f"{backend}_gate_passed"] = float(item["gate"]["passed"])
    return summary


def wandb_metrics(losses: dict, reports: dict[str, dict], score: float, crash_metrics: dict[str, float], transfer_metrics: dict) -> dict[str, float]:
    payload = {"selection_score": float(score)}
    payload.update({f"train/{key}": float(value) for key, value in losses.items()})
    payload.update({f"eval/{key}": value for key, value in summary_report_metrics(reports).items()})
    payload.update({f"selection/{key}": float(value) for key, value in crash_metrics.items()})
    payload.update({f"selection/{key}": value for key, value in numeric_metrics(transfer_metrics).items()})
    return payload
if __name__ == "__main__": main()
