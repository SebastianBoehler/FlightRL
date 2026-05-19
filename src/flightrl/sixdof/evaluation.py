from __future__ import annotations

import numpy as np
import torch

from .env import SixDofCrazyflieEnv
from .policies import SixDofPolicy, teacher_actions
from .tasks import append_task_encoding, parse_task_spec


def checkpoint_tasks(checkpoint: dict, fallback: str = "position_yaw") -> tuple[str, ...]:
    tasks = tuple(checkpoint.get("tasks", ()))
    if tasks:
        return tasks
    return parse_task_spec(str(checkpoint.get("task", fallback)))


def load_policy_from_checkpoint(checkpoint: dict) -> SixDofPolicy:
    model = SixDofPolicy(
        hidden_size=int(checkpoint.get("hidden_size", 128)),
        input_dim=int(checkpoint.get("observation_dim", 28)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def evaluate_policy(
    model: SixDofPolicy,
    tasks: tuple[str, ...],
    *,
    seed: int,
    steps: int = 300,
    num_envs: int = 128,
    use_native_step: bool = False,
) -> dict:
    per_task = {
        task: evaluate_one(model_actions, model, tasks, task, seed + idx, steps, num_envs, use_native_step) for idx, task in enumerate(tasks)
    }
    return aggregate_task_metrics(per_task)


def evaluate_teacher(
    tasks: tuple[str, ...],
    *,
    seed: int,
    steps: int = 300,
    num_envs: int = 128,
    use_native_step: bool = False,
) -> dict:
    per_task = {
        task: evaluate_one(teacher_action, None, tasks, task, seed + idx, steps, num_envs, use_native_step) for idx, task in enumerate(tasks)
    }
    return aggregate_task_metrics(per_task)


def aggregate_task_metrics(per_task: dict[str, dict[str, float]]) -> dict:
    summary = {
        "mean_reward": float(np.mean([metrics["mean_reward"] for metrics in per_task.values()])),
        "mean_position_error_m": float(np.mean([metrics["mean_position_error_m"] for metrics in per_task.values()])),
        "min_clearance_m": float(np.min([metrics["min_clearance_m"] for metrics in per_task.values()])),
        "clearance_p01_m": float(np.min([metrics["clearance_p01_m"] for metrics in per_task.values()])),
        "mean_completed_fraction": float(np.mean([metrics["completed_fraction"] for metrics in per_task.values()])),
        "mean_terminal_fraction": float(np.mean([metrics["terminal_fraction"] for metrics in per_task.values()])),
        "per_task": per_task,
    }
    optional_keys = ("teacher_action_l2_mean", "teacher_action_l2_p95", "action_abs_mean", "action_abs_max", "action_saturation_fraction")
    for key in optional_keys:
        values = [metrics[key] for metrics in per_task.values() if key in metrics]
        if values:
            summary[key] = float(np.mean(values)) if not key.endswith("_max") else float(np.max(values))
    return summary


def evaluate_one(
    action_fn,
    model: SixDofPolicy,
    tasks: tuple[str, ...],
    task: str,
    seed: int,
    steps: int,
    num_envs: int,
    use_native_step: bool,
) -> dict[str, float]:
    env = SixDofCrazyflieEnv(num_envs=num_envs, seed=seed, task=task, use_native_step=use_native_step)
    obs, _ = env.reset(seed=seed)
    task_indices = np.full(env.num_envs, tasks.index(task), dtype=np.int64)
    rewards = []
    min_clearance = []
    action_abs = []
    action_l2 = []
    survived = np.ones(env.num_envs, dtype=bool)
    for _ in range(steps):
        actions = action_fn(model, env, obs, task_indices, tasks, task)
        teacher = teacher_actions(env, task=task)
        action_abs.append(np.abs(actions))
        if model is not None:
            action_l2.append(np.linalg.norm(actions - teacher, axis=1))
        obs, reward, terminals, truncations, _info = env.step(actions)
        rewards.append(reward)
        min_clearance.append(np.min(env.ranges_m[:, :4], axis=1))
        survived &= ~terminals.astype(bool)
    pos_error = np.linalg.norm(env.target_position - env.position, axis=1)
    clearances = np.concatenate(min_clearance)
    result = {
        "mean_reward": float(np.mean(rewards)),
        "mean_position_error_m": float(np.mean(pos_error)),
        "min_clearance_m": float(np.min(clearances)),
        "clearance_p01_m": float(np.quantile(clearances, 0.01)),
        "completed_fraction": float(np.mean(survived)),
        "terminal_fraction": float(1.0 - np.mean(survived)),
        "action_abs_mean": float(np.mean(np.concatenate(action_abs))),
        "action_abs_max": float(np.max(np.concatenate(action_abs))),
        "action_saturation_fraction": float(np.mean(np.concatenate(action_abs) > 0.95)),
    }
    if action_l2:
        action_errors = np.concatenate(action_l2)
        result["teacher_action_l2_mean"] = float(np.mean(action_errors))
        result["teacher_action_l2_p95"] = float(np.quantile(action_errors, 0.95))
    return result


def model_actions(model: SixDofPolicy, _env, obs: np.ndarray, task_indices: np.ndarray, tasks: tuple[str, ...], _task: str) -> np.ndarray:
    model_obs = append_task_encoding(obs, task_indices, len(tasks))
    with torch.no_grad():
        return model(torch.from_numpy(model_obs).float()).cpu().numpy()


def teacher_action(_model, env: SixDofCrazyflieEnv, _obs, _task_indices, _tasks, task: str) -> np.ndarray:
    return teacher_actions(env, task=task)


def gate_status(metrics: dict, *, min_clearance_m: float, min_completed_fraction: float, max_position_error_m: float) -> dict:
    failures = []
    if metrics.get("clearance_p01_m", metrics["min_clearance_m"]) < min_clearance_m:
        failures.append("min_clearance")
    if metrics["mean_completed_fraction"] < min_completed_fraction:
        failures.append("completion")
    if metrics["mean_position_error_m"] > max_position_error_m:
        failures.append("position_error")
    return {"passed": not failures, "failures": failures}
