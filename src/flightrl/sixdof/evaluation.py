from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .circle import circle_orbit_error_from_arrays
from .controller import executed_action_for_controller
from .disturbance import configure_disturbance
from .env import SixDofCrazyflieEnv
from .gates import gate_status
from .observation import augment_observation
from .policies import SixDofPolicy, roll_pitch_from_quat, teacher_actions
from .tasks import append_task_encoding, parse_task_spec
from .yaw import yaw_error_for_task


OPEN_SPACE_CLEARANCE_M = 0.45


@dataclass(frozen=True, slots=True)
class ControllerPolicy:
    model: SixDofPolicy
    controller: str
    residual_scale: float


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


def load_controller_from_checkpoint(checkpoint: dict) -> ControllerPolicy:
    return ControllerPolicy(
        model=load_policy_from_checkpoint(checkpoint),
        controller=str(checkpoint.get("controller", "policy")),
        residual_scale=float(checkpoint.get("residual_scale", 0.0)),
    )


def evaluate_checkpoint_policy(
    checkpoint: dict,
    *,
    seed: int,
    steps: int = 300,
    num_envs: int = 128,
    use_native_step: bool = False,
    eval_tasks: tuple[str, ...] | None = None,
    reset_profile: str | None = None,
    sensor_profile: str | None = None,
    metric_start_step: int = 0,
) -> dict:
    controller = load_controller_from_checkpoint(checkpoint)
    tasks = checkpoint_tasks(checkpoint)
    action_fn = residual_model_actions if controller.controller == "teacher_residual" else model_actions
    observation_mode = str(checkpoint.get("observation_mode", "base"))
    selected_tasks = eval_tasks or tasks
    validate_task_subset(selected_tasks, tasks)
    per_task = {
        task: evaluate_one(action_fn, controller, tasks, task, seed + idx, steps, num_envs, use_native_step, reset_profile, sensor_profile, observation_mode, metric_start_step)
        for idx, task in enumerate(selected_tasks)
    }
    return aggregate_task_metrics(per_task)


def evaluate_policy(
    model: SixDofPolicy,
    tasks: tuple[str, ...],
    *,
    seed: int,
    steps: int = 300,
    num_envs: int = 128,
    use_native_step: bool = False,
    eval_tasks: tuple[str, ...] | None = None,
    reset_profile: str | None = None,
    sensor_profile: str | None = None,
    observation_mode: str = "base",
    metric_start_step: int = 0,
) -> dict:
    selected_tasks = eval_tasks or tasks
    validate_task_subset(selected_tasks, tasks)
    per_task = {
        task: evaluate_one(model_actions, model, tasks, task, seed + idx, steps, num_envs, use_native_step, reset_profile, sensor_profile, observation_mode, metric_start_step)
        for idx, task in enumerate(selected_tasks)
    }
    return aggregate_task_metrics(per_task)


def evaluate_teacher(
    tasks: tuple[str, ...],
    *,
    seed: int,
    steps: int = 300,
    num_envs: int = 128,
    use_native_step: bool = False,
    reset_profile: str | None = None,
    sensor_profile: str | None = None,
    metric_start_step: int = 0,
) -> dict:
    per_task = {
        task: evaluate_one(teacher_action, None, tasks, task, seed + idx, steps, num_envs, use_native_step, reset_profile, sensor_profile, "base", metric_start_step)
        for idx, task in enumerate(tasks)
    }
    return aggregate_task_metrics(per_task)


def aggregate_task_metrics(per_task: dict[str, dict[str, float]]) -> dict:
    summary = {
        "mean_reward": float(np.mean([metrics["mean_reward"] for metrics in per_task.values()])),
        "mean_position_error_m": float(np.mean([metrics["mean_position_error_m"] for metrics in per_task.values()])),
        "mean_yaw_error_rad": float(np.mean([metrics["mean_yaw_error_rad"] for metrics in per_task.values()])),
        "yaw_error_p95_rad": float(np.max([metrics["yaw_error_p95_rad"] for metrics in per_task.values()])),
        "settled_yaw_error_p95_rad": float(np.max([metrics["settled_yaw_error_p95_rad"] for metrics in per_task.values()])),
        "min_clearance_m": float(np.min([metrics["min_clearance_m"] for metrics in per_task.values()])),
        "clearance_p01_m": float(np.min([metrics["clearance_p01_m"] for metrics in per_task.values()])),
        "mean_completed_fraction": float(np.mean([metrics["completed_fraction"] for metrics in per_task.values()])),
        "mean_survival_fraction": float(np.mean([metrics["survival_fraction"] for metrics in per_task.values()])),
        "mean_terminal_fraction": float(np.mean([metrics["terminal_fraction"] for metrics in per_task.values()])),
        "per_task": per_task,
    }
    optional_keys = (
        "teacher_action_l2_mean",
        "teacher_action_l2_p95",
        "action_abs_mean",
        "action_abs_max",
        "action_saturation_fraction",
        "horizontal_speed_p95_m_s",
        "horizontal_speed_max_m_s",
        "open_space_horizontal_speed_p95_m_s",
        "open_space_horizontal_speed_max_m_s",
        "tilt_p95_deg",
        "tilt_max_deg",
    )
    for key in optional_keys:
        values = [metrics[key] for metrics in per_task.values() if key in metrics]
        if values:
            summary[key] = float(np.max(values)) if "_max" in key else float(np.mean(values))
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
    reset_profile: str | None,
    sensor_profile: str | None,
    observation_mode: str,
    metric_start_step: int = 0,
    physics_profile: str | None = None,
    domain_randomization: str | None = None,
    disturbance_profile: str | None = None,
) -> dict[str, float]:
    env = SixDofCrazyflieEnv(
        num_envs=num_envs,
        seed=seed,
        task=task,
        use_native_step=use_native_step,
        reset_profile=reset_profile,
        sensor_profile=sensor_profile,
        physics_profile=physics_profile,
        domain_randomization=domain_randomization,
    )
    configure_disturbance(env, disturbance_profile)
    obs, _ = env.reset(seed=seed)
    task_indices = np.full(env.num_envs, tasks.index(task), dtype=np.int64)
    rewards = []
    min_clearance = []
    action_abs = []
    action_l2 = []
    yaw_errors = []
    horizontal_speed = []
    open_space_horizontal_speed = []
    tilt = []
    survived = np.ones(env.num_envs, dtype=bool)
    ever_close = np.min(env.ranges_m[:, :4], axis=1) < OPEN_SPACE_CLEARANCE_M
    alive_samples = []
    previous_obs = None
    previous_action = np.zeros((env.num_envs, 4), dtype=np.float32)
    fresh = np.ones(env.num_envs, dtype=bool)
    for _ in range(steps):
        model_obs = append_task_encoding(obs.copy(), task_indices, len(tasks))
        if previous_obs is None:
            previous_obs = model_obs.copy()
        previous_obs[fresh] = model_obs[fresh]
        policy_obs = augment_observation(model_obs, previous_obs, previous_action, observation_mode)
        actions = action_fn(model, env, policy_obs, task_indices, tasks, task)
        teacher = teacher_actions(env, task=task)
        action_abs.append(np.abs(actions))
        if model is not None:
            action_l2.append(np.linalg.norm(actions - teacher, axis=1))
        obs, reward, terminals, truncations, _info = env.step(actions)
        previous_obs = model_obs.copy()
        previous_action = actions.copy()
        rewards.append(reward)
        current_clearance = np.min(env.ranges_m[:, :4], axis=1)
        current_horizontal_speed = np.linalg.norm(env.velocity[:, :2], axis=1)
        min_clearance.append(current_clearance)
        yaw_errors.append(yaw_error_for_task(env, task))
        horizontal_speed.append(current_horizontal_speed)
        ever_close |= current_clearance < OPEN_SPACE_CLEARANCE_M
        free_space = (current_clearance >= OPEN_SPACE_CLEARANCE_M) & ~ever_close
        open_space_horizontal_speed.append(current_horizontal_speed[free_space])
        roll, pitch = roll_pitch_from_quat(env.quaternion)
        tilt.append(np.rad2deg(np.maximum(np.abs(roll), np.abs(pitch))))
        survived &= ~terminals.astype(bool)
        alive_samples.append(survived.astype(np.float32))
        fresh = (terminals | truncations).astype(bool)
        previous_action[fresh] = 0.0
    pos_error = position_error_for_task(env, task)
    yaw_error = yaw_error_for_task(env, task)
    clearances = np.concatenate(min_clearance)
    yaw_error_samples = np.concatenate(yaw_errors)
    horizontal_speed_samples = np.concatenate(horizontal_speed)
    open_space_speed_samples = np.concatenate([sample for sample in open_space_horizontal_speed if sample.size]) if any(sample.size for sample in open_space_horizontal_speed) else np.asarray([0.0])
    tilt_samples = np.concatenate(tilt)
    settled_yaw = np.concatenate(yaw_errors[max(0, min(metric_start_step, len(yaw_errors) - 1)) :])
    result = {
        "mean_reward": float(np.mean(rewards)),
        "mean_position_error_m": float(np.mean(pos_error)),
        "mean_yaw_error_rad": float(np.mean(yaw_error)),
        "yaw_error_p95_rad": float(np.quantile(yaw_error_samples, 0.95)),
        "settled_yaw_error_p95_rad": float(np.quantile(settled_yaw, 0.95)),
        "metric_start_step": float(metric_start_step),
        "min_clearance_m": float(np.min(clearances)),
        "clearance_p01_m": float(np.quantile(clearances, 0.01)),
        "completed_fraction": float(np.mean(survived)),
        "survival_fraction": float(np.mean(np.concatenate(alive_samples))),
        "terminal_fraction": float(1.0 - np.mean(survived)),
        "action_abs_mean": float(np.mean(np.concatenate(action_abs))),
        "action_abs_max": float(np.max(np.concatenate(action_abs))),
        "action_saturation_fraction": float(np.mean(np.concatenate(action_abs) > 0.95)),
        "horizontal_speed_p95_m_s": float(np.quantile(horizontal_speed_samples, 0.95)),
        "horizontal_speed_max_m_s": float(np.max(horizontal_speed_samples)),
        "open_space_horizontal_speed_p95_m_s": float(np.quantile(open_space_speed_samples, 0.95)),
        "open_space_horizontal_speed_max_m_s": float(np.max(open_space_speed_samples)),
        "tilt_p95_deg": float(np.quantile(tilt_samples, 0.95)),
        "tilt_max_deg": float(np.max(tilt_samples)),
    }
    if action_l2:
        action_errors = np.concatenate(action_l2)
        result["teacher_action_l2_mean"] = float(np.mean(action_errors))
        result["teacher_action_l2_p95"] = float(np.quantile(action_errors, 0.95))
    return result


def model_actions(model: SixDofPolicy, _env, obs: np.ndarray, task_indices: np.ndarray, tasks: tuple[str, ...], _task: str) -> np.ndarray:
    if isinstance(model, ControllerPolicy):
        model = model.model
    with torch.no_grad():
        return model(torch.from_numpy(obs).float()).cpu().numpy()


def residual_model_actions(controller: ControllerPolicy, env, obs: np.ndarray, _task_indices: np.ndarray, _tasks: tuple[str, ...], task: str) -> np.ndarray:
    base = teacher_actions(env, task=task)
    with torch.no_grad():
        residual = controller.model(torch.from_numpy(obs).float()).cpu().numpy()
    return executed_action_for_controller("teacher_residual", residual, base, controller.residual_scale)


def teacher_action(_model, env: SixDofCrazyflieEnv, _obs, _task_indices, _tasks, task: str) -> np.ndarray:
    return teacher_actions(env, task=task)


def position_error_for_task(env: SixDofCrazyflieEnv, task: str) -> np.ndarray:
    if task == "circle":
        return circle_orbit_error_from_arrays(env.position, env.target_position)
    return np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)


def validate_task_subset(selected_tasks: tuple[str, ...], policy_tasks: tuple[str, ...]) -> None:
    missing = [task for task in selected_tasks if task not in policy_tasks]
    if missing:
        raise ValueError(f"selected task(s) not present in checkpoint: {', '.join(missing)}")
