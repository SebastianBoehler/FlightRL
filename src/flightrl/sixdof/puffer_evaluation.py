from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from flightrl.mujoco import MuJoCoCrazyflieEnv, is_mujoco_available

from .disturbance import configure_disturbance
from .evaluation import OPEN_SPACE_CLEARANCE_M, aggregate_task_metrics, evaluate_one, gate_status
from .physics import SixDofPhysicsProfile
from .puffer_observation import scale_previous_action_observation
from .policies import roll_pitch_from_quat
from .tasks import append_task_encoding
from .yaw import yaw_error_for_task


@dataclass(frozen=True, slots=True)
class PufferEvalConfig:
    task: str = "obstacle_avoidance"
    backend: str = "both"
    steps: int = 300
    num_envs: int = 128
    seed: int = 707
    reset_profile: str = "obstacle_close_live"
    sensor_profile: str | None = None
    physics_profile: str | SixDofPhysicsProfile | None = None
    domain_randomization: str | None = None
    disturbance_profile: str | None = None
    min_clearance_m: float = 0.08
    min_completed_fraction: float = 0.90
    max_position_error_m: float = 1.00
    max_horizontal_speed_p95_m_s: float = 1.50
    max_open_space_horizontal_speed_p95_m_s: float | None = None
    max_tilt_p95_deg: float = 35.0
    previous_action_observation_scale: float = 1.0


def evaluate_puffer_backends(policy, config: PufferEvalConfig) -> dict[str, dict]:
    reports = {}
    if config.backend in {"python", "both"}:
        reports["python"] = evaluate_puffer_python(policy, config)
    if config.backend in {"mujoco", "both"}:
        reports["mujoco"] = evaluate_puffer_mujoco(policy, config)
    return reports


def evaluate_puffer_python(policy, config: PufferEvalConfig) -> dict:
    def action_fn(model, env, obs, task_indices, tasks, task):
        scaled = scale_previous_action_observation(obs, config.previous_action_observation_scale)
        return puffer_actions(model, env, scaled, task_indices, tasks, task)

    per_task = {
        config.task: evaluate_one(
            action_fn,
            policy,
            (config.task,),
            config.task,
            config.seed,
            config.steps,
            config.num_envs,
            False,
            config.reset_profile,
            config.sensor_profile,
            "base",
            physics_profile=config.physics_profile,
            domain_randomization=config.domain_randomization,
            disturbance_profile=config.disturbance_profile,
        )
    }
    return puffer_gate_report("python", aggregate_task_metrics(per_task), config)


def evaluate_puffer_mujoco(policy, config: PufferEvalConfig) -> dict:
    if not is_mujoco_available():
        return {"status": "missing_mujoco", "gate": {"passed": False, "failures": ["missing_mujoco"]}}
    env = MuJoCoCrazyflieEnv(
        num_envs=config.num_envs,
        seed=config.seed + 1000,
        task=config.task,
        reset_profile=config.reset_profile,
        sensor_profile=config.sensor_profile,
        physics_profile=config.physics_profile,
    )
    configure_disturbance(env, config.disturbance_profile)
    obs, _ = env.reset(seed=config.seed + 1000)
    rewards, clearances, action_abs, yaw_errors, horizontal_speed, open_space_horizontal_speed, tilt = [], [], [], [], [], [], []
    survived = np.ones(config.num_envs, dtype=bool)
    ever_close = np.min(env.ranges_m[:, :4], axis=1) < OPEN_SPACE_CLEARANCE_M
    alive_samples = []
    task_indices = np.zeros(config.num_envs, dtype=np.int64)
    for _ in range(config.steps):
        policy_obs = append_task_encoding(obs.copy(), task_indices, 1)
        policy_obs = scale_previous_action_observation(policy_obs, config.previous_action_observation_scale)
        actions = puffer_actions(policy, env, policy_obs, task_indices, (config.task,), config.task)
        obs, reward, terminals, truncations, _ = env.step(actions)
        done = terminals | truncations
        rewards.append(reward.copy())
        clearance = np.min(env.ranges_m[:, :4], axis=1)
        speed = np.linalg.norm(env.velocity[:, :2], axis=1)
        clearances.append(clearance)
        action_abs.append(np.abs(actions))
        yaw_errors.append(yaw_error_for_task(env, config.task))
        horizontal_speed.append(speed)
        ever_close |= clearance < OPEN_SPACE_CLEARANCE_M
        free_space = (clearance >= OPEN_SPACE_CLEARANCE_M) & ~ever_close
        open_space_horizontal_speed.append(speed[free_space])
        roll, pitch = roll_pitch_from_quat(env.quaternion)
        tilt.append(np.rad2deg(np.maximum(np.abs(roll), np.abs(pitch))))
        survived &= ~terminals.astype(bool)
        alive_samples.append(survived.astype(np.float32))
        if np.any(done):
            obs = env.reset_done(done).copy()
    return puffer_gate_report("mujoco", metrics_from_samples(env, rewards, clearances, action_abs, yaw_errors, horizontal_speed, open_space_horizontal_speed, tilt, survived, alive_samples), config)


def puffer_actions(policy, _env, obs: np.ndarray, _task_indices, _tasks, _task) -> np.ndarray:
    if obs.shape[1] != policy.metadata.observation_dim:
        raise ValueError(f"Puffer checkpoint expects obs_dim={policy.metadata.observation_dim}, got {obs.shape[1]}")
    with torch.no_grad():
        return policy(torch.from_numpy(obs).float()).cpu().numpy().astype(np.float32)


def metrics_from_samples(env, rewards, clearances, action_abs, yaw_errors, horizontal_speed, open_space_horizontal_speed, tilt, survived, alive_samples) -> dict:
    clear = np.concatenate(clearances)
    actions = np.concatenate(action_abs)
    speed = np.concatenate(horizontal_speed)
    open_space_speed = np.concatenate([sample for sample in open_space_horizontal_speed if sample.size]) if any(sample.size for sample in open_space_horizontal_speed) else np.asarray([0.0])
    tilt_samples = np.concatenate(tilt)
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
        "horizontal_speed_p95_m_s": float(np.quantile(speed, 0.95)),
        "horizontal_speed_max_m_s": float(np.max(speed)),
        "open_space_horizontal_speed_p95_m_s": float(np.quantile(open_space_speed, 0.95)),
        "open_space_horizontal_speed_max_m_s": float(np.max(open_space_speed)),
        "tilt_p95_deg": float(np.quantile(tilt_samples, 0.95)),
        "tilt_max_deg": float(np.max(tilt_samples)),
    }


def puffer_gate_report(backend: str, metrics: dict, config: PufferEvalConfig) -> dict:
    gate = gate_status(
        metrics,
        min_clearance_m=config.min_clearance_m,
        min_completed_fraction=config.min_completed_fraction,
        max_position_error_m=config.max_position_error_m,
        max_horizontal_speed_p95_m_s=config.max_horizontal_speed_p95_m_s,
        max_open_space_horizontal_speed_p95_m_s=config.max_open_space_horizontal_speed_p95_m_s,
        max_tilt_p95_deg=config.max_tilt_p95_deg,
    )
    return {"status": "ok", "backend": backend, "gate": gate, "metrics": metrics}
