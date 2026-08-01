from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flightrl.mujoco import MuJoCoCrazyflieEnv, is_mujoco_available
from flightrl.sixdof import checkpoint_tasks, gate_status, load_controller_from_checkpoint, teacher_actions
from flightrl.sixdof.evaluation import position_error_for_task
from flightrl.sixdof.controller import executed_action_for_controller
from flightrl.sixdof.observation import augment_observation
from flightrl.sixdof.tasks import append_task_encoding, parse_task_spec
from flightrl.sixdof.yaw import yaw_error_for_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 6-DoF teacher/checkpoint behavior in the MuJoCo backend.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--reset-profile", default="obstacle_close_live")
    parser.add_argument("--sensor-profile", default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-clearance-m", type=float, default=0.08)
    parser.add_argument("--min-completed-fraction", type=float, default=0.90)
    parser.add_argument("--max-position-error-m", type=float, default=1.00)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not is_mujoco_available():
        report = {"status": "missing_mujoco", "passed": False}
    else:
        report = evaluate(args)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_report(report) + "\n")
    print(f"mujoco_report={output}")
    print(f"status={report['status']} passed={report.get('gate', {}).get('passed', False)}")


def evaluate(args: argparse.Namespace) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu") if args.checkpoint else None
    tasks = checkpoint_tasks(checkpoint) if checkpoint else parse_task_spec(args.task)
    task = parse_task_spec(args.task)[0]
    if task not in tasks:
        raise SystemExit(f"task {task!r} not present in checkpoint tasks {tasks}")
    env = MuJoCoCrazyflieEnv(num_envs=args.num_envs, seed=args.seed, task=task, reset_profile=args.reset_profile, sensor_profile=args.sensor_profile)
    obs, _ = env.reset(seed=args.seed)
    controller = load_controller_from_checkpoint(checkpoint) if checkpoint and not args.teacher else None
    observation_mode = str(checkpoint.get("observation_mode", "base")) if checkpoint else "base"
    task_indices = np.full(args.num_envs, tasks.index(task), dtype=np.int64)
    previous_obs = None
    previous_action = np.zeros((args.num_envs, 4), dtype=np.float32)
    rewards, clearances, action_abs, action_l2, yaw_errors = [], [], [], [], []
    survived = np.ones(args.num_envs, dtype=bool)
    alive_samples = []
    for _ in range(args.steps):
        teacher = teacher_actions(env, task=task)
        if args.teacher:
            actions = teacher
        else:
            assert controller is not None
            model_obs = append_task_encoding(obs.copy(), task_indices, len(tasks))
            if previous_obs is None:
                previous_obs = model_obs.copy()
            policy_obs = augment_observation(model_obs, previous_obs, previous_action, observation_mode)
            with torch.no_grad():
                raw = controller.model(torch.from_numpy(policy_obs).float()).cpu().numpy()
            actions = executed_action_for_controller(controller.controller, raw, teacher, controller.residual_scale)
            previous_obs = model_obs.copy()
            previous_action = actions.copy()
            action_l2.append(np.linalg.norm(actions - teacher, axis=1))
        obs, reward, terminals, truncations, _ = env.step(actions)
        done = terminals | truncations
        rewards.append(reward.copy())
        clearances.append(np.min(env.ranges_m[:, :4], axis=1))
        action_abs.append(np.abs(actions))
        yaw_errors.append(yaw_error_for_task(env, task))
        survived &= ~terminals.astype(bool)
        alive_samples.append(survived.astype(np.float32))
        if np.any(done):
            obs = env.reset_done(done).copy()
            previous_action[done.astype(bool)] = 0.0
    metrics = metrics_from_samples(env, task, rewards, clearances, action_abs, action_l2, yaw_errors, survived, alive_samples)
    gate = gate_status(
        metrics,
        min_clearance_m=args.min_clearance_m,
        min_completed_fraction=args.min_completed_fraction,
        max_position_error_m=args.max_position_error_m,
    )
    return {
        "status": "ok",
        "backend": "mujoco",
        "checkpoint": args.checkpoint,
        "controller": "teacher" if args.teacher else checkpoint.get("controller", "checkpoint"),
        "tasks": [task],
        "steps": args.steps,
        "num_envs": args.num_envs,
        "reset_profile": args.reset_profile,
        "sensor_profile": args.sensor_profile,
        "gate": gate,
        "metrics": metrics,
        "safety": "MuJoCo validation only; passing this report does not approve live hardware deployment.",
    }


def metrics_from_samples(env, task, rewards, clearances, action_abs, action_l2, yaw_errors, survived, alive_samples) -> dict:
    pos_error = position_error_for_task(env, task)
    clear = np.concatenate(clearances)
    action_values = np.concatenate(action_abs)
    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "mean_position_error_m": float(np.mean(pos_error)),
        "mean_yaw_error_rad": float(np.mean(yaw_errors[-1])),
        "yaw_error_p95_rad": float(np.quantile(np.concatenate(yaw_errors), 0.95)),
        "min_clearance_m": float(np.min(clear)),
        "clearance_p01_m": float(np.quantile(clear, 0.01)),
        "mean_completed_fraction": float(np.mean(survived)),
        "mean_survival_fraction": float(np.mean(np.concatenate(alive_samples))),
        "mean_terminal_fraction": float(1.0 - np.mean(survived)),
        "action_abs_mean": float(np.mean(action_values)),
        "action_abs_max": float(np.max(action_values)),
        "action_saturation_fraction": float(np.mean(action_values > 0.95)),
    }
    if action_l2:
        errors = np.concatenate(action_l2)
        metrics["teacher_action_l2_mean"] = float(np.mean(errors))
        metrics["teacher_action_l2_p95"] = float(np.quantile(errors, 0.95))
    return metrics


def render_report(report: dict) -> str:
    if report["status"] != "ok":
        return "# MuJoCo Six-DoF Evaluation\n\nMuJoCo is not installed."
    m = report["metrics"]
    return "\n".join(
        [
            "# MuJoCo Six-DoF Evaluation",
            "",
            f"- Controller: `{report['controller']}`",
            f"- Passed: `{report['gate']['passed']}`",
            f"- Failures: `{', '.join(report['gate']['failures']) or 'none'}`",
            f"- Completed: `{m['mean_completed_fraction']:.3f}`",
            f"- Position error: `{m['mean_position_error_m']:.3f}` m",
            f"- Clearance p01: `{m['clearance_p01_m']:.3f}` m",
            f"- Min clearance: `{m['min_clearance_m']:.3f}` m",
            f"- Teacher L2: `{m.get('teacher_action_l2_mean', 0.0):.4f}`",
            "",
            report["safety"],
        ]
    )


if __name__ == "__main__":
    main()
