from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flightrl.mujoco import MuJoCoCrazyflieEnv, is_mujoco_available
from flightrl.sixdof.evaluation import aggregate_task_metrics, evaluate_one, gate_status
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy
from flightrl.sixdof.tasks import append_task_encoding, parse_task_spec
from flightrl.sixdof.yaw import yaw_error_for_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PufferLib six-DoF checkpoint in Python and MuJoCo gates.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="obstacle_avoidance")
    parser.add_argument("--backend", choices=("python", "mujoco", "both"), default="both")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--reset-profile", default="obstacle_close_live")
    parser.add_argument("--sensor-profile", default=None)
    parser.add_argument("--min-clearance-m", type=float, default=0.08)
    parser.add_argument("--min-completed-fraction", type=float, default=0.90)
    parser.add_argument("--max-position-error-m", type=float, default=1.00)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    policy = load_puffer_sixdof_policy(args.checkpoint)
    tasks = parse_task_spec(args.task)
    reports = {}
    if args.backend in {"python", "both"}:
        reports["python"] = evaluate_python(policy, args, tasks)
    if args.backend in {"mujoco", "both"}:
        reports["mujoco"] = evaluate_mujoco(policy, args, tasks[0])
    report = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "policy": {
            "type": "pufferlib_mlp_mean",
            "observation_dim": policy.metadata.observation_dim,
            "hidden_size": policy.metadata.hidden_size,
            "action_dim": policy.metadata.action_dim,
            "num_layers": policy.metadata.num_layers,
        },
        "tasks": list(tasks),
        "steps": args.steps,
        "num_envs": args.num_envs,
        "reset_profile": args.reset_profile,
        "sensor_profile": args.sensor_profile,
        "thresholds": thresholds(args),
        "reports": reports,
        "passed": all(item.get("gate", {}).get("passed", False) for item in reports.values()),
        "safety": "Offline simulation gate only; passing this report does not approve live hardware deployment.",
    }
    write_report(report, Path(args.output))
    if args.fail_on_gate and not report["passed"]:
        raise SystemExit(2)


def evaluate_python(policy, args: argparse.Namespace, tasks: tuple[str, ...]) -> dict:
    per_task = {
        task: evaluate_one(
            puffer_actions,
            policy,
            tasks,
            task,
            args.seed + idx,
            args.steps,
            args.num_envs,
            False,
            args.reset_profile,
            args.sensor_profile,
            "base",
        )
        for idx, task in enumerate(tasks)
    }
    metrics = aggregate_task_metrics(per_task)
    return gate_report("python", metrics, args)


def evaluate_mujoco(policy, args: argparse.Namespace, task: str) -> dict:
    if not is_mujoco_available():
        return {"status": "missing_mujoco", "gate": {"passed": False, "failures": ["missing_mujoco"]}}
    env = MuJoCoCrazyflieEnv(num_envs=args.num_envs, seed=args.seed + 1000, task=task, reset_profile=args.reset_profile, sensor_profile=args.sensor_profile)
    obs, _ = env.reset(seed=args.seed + 1000)
    rewards, clearances, action_abs, yaw_errors = [], [], [], []
    survived = np.ones(args.num_envs, dtype=bool)
    alive_samples = []
    task_indices = np.zeros(args.num_envs, dtype=np.int64)
    for _ in range(args.steps):
        policy_obs = append_task_encoding(obs.copy(), task_indices, 1)
        actions = puffer_actions(policy, env, policy_obs, task_indices, (task,), task)
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
    metrics = metrics_from_samples(env, rewards, clearances, action_abs, yaw_errors, survived, alive_samples)
    return gate_report("mujoco", metrics, args)


def puffer_actions(policy, _env, obs: np.ndarray, _task_indices, _tasks, _task) -> np.ndarray:
    if obs.shape[1] != policy.metadata.observation_dim:
        raise ValueError(f"Puffer checkpoint expects obs_dim={policy.metadata.observation_dim}, got {obs.shape[1]}")
    with torch.no_grad():
        return policy(torch.from_numpy(obs).float()).cpu().numpy().astype(np.float32)


def metrics_from_samples(env, rewards, clearances, action_abs, yaw_errors, survived, alive_samples) -> dict:
    clear = np.concatenate(clearances)
    actions = np.concatenate(action_abs)
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_position_error_m": float(np.mean(np.linalg.norm(env.target_position - env.position, axis=1))),
        "mean_yaw_error_rad": float(np.mean(yaw_errors[-1])),
        "yaw_error_p95_rad": float(np.quantile(np.concatenate(yaw_errors), 0.95)),
        "min_clearance_m": float(np.min(clear)),
        "clearance_p01_m": float(np.quantile(clear, 0.01)),
        "mean_completed_fraction": float(np.mean(survived)),
        "mean_survival_fraction": float(np.mean(np.concatenate(alive_samples))),
        "mean_terminal_fraction": float(1.0 - np.mean(survived)),
        "action_abs_mean": float(np.mean(actions)),
        "action_abs_max": float(np.max(actions)),
        "action_saturation_fraction": float(np.mean(actions > 0.95)),
    }


def gate_report(backend: str, metrics: dict, args: argparse.Namespace) -> dict:
    gate = gate_status(
        metrics,
        min_clearance_m=args.min_clearance_m,
        min_completed_fraction=args.min_completed_fraction,
        max_position_error_m=args.max_position_error_m,
    )
    return {"status": "ok", "backend": backend, "gate": gate, "metrics": metrics}


def thresholds(args: argparse.Namespace) -> dict:
    return {
        "min_clearance_m": args.min_clearance_m,
        "min_completed_fraction": args.min_completed_fraction,
        "max_position_error_m": args.max_position_error_m,
    }


def write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"puffer_eval={output}")
    print(f"passed={report['passed']}")


def render_markdown(report: dict) -> str:
    lines = ["# Puffer Six-DoF Evaluation", "", f"Passed: `{report['passed']}`", ""]
    for name, item in report["reports"].items():
        if item["status"] != "ok":
            lines.extend([f"## {name}", "", f"Status: `{item['status']}`", ""])
            continue
        metrics = item["metrics"]
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Gate: `{item['gate']['passed']}`",
                f"- Failures: `{', '.join(item['gate']['failures']) or 'none'}`",
                f"- Reward: `{metrics['mean_reward']:.3f}`",
                f"- Position error: `{metrics['mean_position_error_m']:.3f}` m",
                f"- Clearance p01: `{metrics['clearance_p01_m']:.3f}` m",
                f"- Survival: `{metrics['mean_survival_fraction']:.3f}`",
                "",
            ]
        )
    lines.append(report["safety"])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
