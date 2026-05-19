from __future__ import annotations

import numpy as np
import torch

from .env import SixDofCrazyflieEnv
from .policies import SixDofPolicy
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
        task: evaluate_one(model, tasks, task, seed + idx, steps, num_envs, use_native_step) for idx, task in enumerate(tasks)
    }
    return {
        "mean_reward": float(np.mean([metrics["mean_reward"] for metrics in per_task.values()])),
        "mean_position_error_m": float(np.mean([metrics["mean_position_error_m"] for metrics in per_task.values()])),
        "min_clearance_m": float(np.min([metrics["min_clearance_m"] for metrics in per_task.values()])),
        "mean_completed_fraction": float(np.mean([metrics["completed_fraction"] for metrics in per_task.values()])),
        "per_task": per_task,
    }


def evaluate_one(
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
    completed = np.ones(env.num_envs, dtype=bool)
    for _ in range(steps):
        model_obs = append_task_encoding(obs, task_indices, len(tasks))
        with torch.no_grad():
            actions = model(torch.from_numpy(model_obs).float()).cpu().numpy()
        obs, reward, terminals, truncations, _info = env.step(actions)
        rewards.append(reward)
        min_clearance.append(np.min(env.ranges_m[:, :4], axis=1))
        completed &= ~(terminals.astype(bool) | truncations.astype(bool))
    pos_error = np.linalg.norm(env.target_position - env.position, axis=1)
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_position_error_m": float(np.mean(pos_error)),
        "min_clearance_m": float(np.min(min_clearance)),
        "completed_fraction": float(np.mean(completed)),
    }


def gate_status(metrics: dict, *, min_clearance_m: float, min_completed_fraction: float, max_position_error_m: float) -> dict:
    failures = []
    if metrics["min_clearance_m"] < min_clearance_m:
        failures.append("min_clearance")
    if metrics["mean_completed_fraction"] < min_completed_fraction:
        failures.append("completion")
    if metrics["mean_position_error_m"] > max_position_error_m:
        failures.append("position_error")
    return {"passed": not failures, "failures": failures}
