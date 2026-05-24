from __future__ import annotations

import numpy as np
import torch

from .controller import executed_action_for_controller
from .env import SixDofCrazyflieEnv, quat_to_yaw, wrap_angle
from .evaluation import ControllerPolicy
from .observation import augment_observation
from .policies import SixDofPolicy, teacher_actions
from .tasks import append_task_encoding


def diagnose_controller(
    model: SixDofPolicy | ControllerPolicy | None,
    policy_tasks: tuple[str, ...],
    *,
    task: str,
    reset_profile: str,
    seed: int,
    steps: int,
    num_envs: int,
    observation_mode: str = "base",
    use_native_step: bool = False,
    bins: int = 8,
) -> dict:
    env = SixDofCrazyflieEnv(num_envs=num_envs, seed=seed, task=task, reset_profile=reset_profile, use_native_step=use_native_step)
    obs, _ = env.reset(seed=seed)
    task_indices = np.full(env.num_envs, policy_tasks.index(task), dtype=np.int64)
    survived = np.ones(env.num_envs, dtype=bool)
    previous_obs = None
    previous_action = np.zeros((env.num_envs, 4), dtype=np.float32)
    fresh = np.ones(env.num_envs, dtype=bool)
    samples = []
    for step in range(steps):
        model_obs = append_task_encoding(obs.copy(), task_indices, len(policy_tasks))
        if previous_obs is None:
            previous_obs = model_obs.copy()
        previous_obs[fresh] = model_obs[fresh]
        policy_obs = augment_observation(model_obs, previous_obs, previous_action, observation_mode)
        actions = controller_actions(model, env, policy_obs, task)
        obs, _reward, terminals, truncations, _info = env.step(actions)
        survived &= ~terminals.astype(bool)
        samples.append(step_metrics(step, env, actions, survived))
        previous_obs = model_obs.copy()
        previous_action = actions.copy()
        fresh = (terminals | truncations).astype(bool)
        previous_action[fresh] = 0.0
    return {
        "task": task,
        "reset_profile": reset_profile,
        "steps": steps,
        "num_envs": num_envs,
        "seed": seed,
        "final": samples[-1] if samples else {},
        "phase_summary": phase_summary(samples),
        "timeline": bin_samples(samples, bins),
    }


def controller_actions(model: SixDofPolicy | ControllerPolicy | None, env: SixDofCrazyflieEnv, policy_obs: np.ndarray, task: str) -> np.ndarray:
    if model is None:
        return teacher_actions(env, task=task)
    if isinstance(model, ControllerPolicy):
        base = teacher_actions(env, task=task)
        with torch.no_grad():
            residual = model.model(torch.from_numpy(policy_obs).float()).cpu().numpy()
        return executed_action_for_controller(model.controller, residual, base, model.residual_scale)
    with torch.no_grad():
        return model(torch.from_numpy(policy_obs).float()).cpu().numpy()


def step_metrics(step: int, env: SixDofCrazyflieEnv, actions: np.ndarray, survived: np.ndarray) -> dict[str, float]:
    position_error = np.linalg.norm(env.target_position - env.position, axis=1)
    clearance = np.min(env.ranges_m[:, :4], axis=1)
    yaw_error = np.abs(wrap_angle(env.target_yaw - quat_to_yaw(env.quaternion)))
    speed = np.linalg.norm(env.velocity, axis=1)
    action_abs = np.abs(actions)
    return {
        "step": float(step),
        "position_error_mean_m": float(np.mean(position_error)),
        "position_error_p95_m": float(np.quantile(position_error, 0.95)),
        "clearance_p01_m": float(np.quantile(clearance, 0.01)),
        "yaw_error_mean_rad": float(np.mean(yaw_error)),
        "yaw_error_p95_rad": float(np.quantile(yaw_error, 0.95)),
        "speed_mean_m_s": float(np.mean(speed)),
        "survival_fraction": float(np.mean(survived)),
        "action_abs_mean": float(np.mean(action_abs)),
        "action_saturation_fraction": float(np.mean(action_abs > 0.95)),
    }


def bin_samples(samples: list[dict[str, float]], bins: int) -> list[dict[str, float]]:
    if not samples:
        return []
    bins = max(1, min(int(bins), len(samples)))
    edges = np.linspace(0, len(samples), bins + 1, dtype=int)
    binned = []
    for start, end in zip(edges[:-1], edges[1:]):
        chunk = samples[start:end]
        if not chunk:
            continue
        keys = [key for key in chunk[0] if key != "step"]
        row = {"step_start": chunk[0]["step"], "step_end": chunk[-1]["step"]}
        row.update({key: float(np.mean([item[key] for item in chunk])) for key in keys})
        binned.append(row)
    return binned


def summarize_diagnostics(records: list[dict]) -> dict:
    blocked = []
    for record in records:
        final = record.get("final", {})
        if final.get("survival_fraction", 0.0) < 0.9:
            blocked.append({"task": record["task"], "profile": record["reset_profile"], "reason": "survival"})
        elif final.get("position_error_mean_m", 0.0) > 1.0:
            blocked.append({"task": record["task"], "profile": record["reset_profile"], "reason": "position_error"})
        elif final.get("clearance_p01_m", 0.0) < 0.08:
            blocked.append({"task": record["task"], "profile": record["reset_profile"], "reason": "clearance"})
    return {"records": len(records), "blocked": blocked, "blocked_count": len(blocked)}


def phase_summary(samples: list[dict[str, float]]) -> dict:
    return {"full": summarize_phase(samples), "settled_half": summarize_phase(samples[len(samples) // 2 :])}


def summarize_phase(samples: list[dict[str, float]]) -> dict:
    if not samples:
        return {}
    return {
        "steps": len(samples),
        "yaw_error_mean_rad": float(np.mean([item["yaw_error_mean_rad"] for item in samples])),
        "yaw_error_p95_rad": float(np.max([item["yaw_error_p95_rad"] for item in samples])),
        "position_error_mean_m": float(np.mean([item["position_error_mean_m"] for item in samples])),
        "clearance_p01_m": float(np.min([item["clearance_p01_m"] for item in samples])),
        "survival_fraction": float(samples[-1]["survival_fraction"]),
    }
