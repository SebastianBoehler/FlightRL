from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .env import ACTION_DIM, SixDofCrazyflieEnv
from .policies import teacher_actions
from .puffer_policy import PufferSixDofPolicy
from .puffer_observation import scale_previous_action_observation
from .puffer_rewards import (
    aggressive_drift_recovery_reward,
    drift_recovery_reward,
    hover_transfer_reward,
    precontact_drift_brake_reward,
    precontact_transfer_reward,
    startup_drift_recovery_reward,
)
from .puffer_transfer_loss import puffer_crash_replay_mse, puffer_transfer_replay_mse, transfer_sign_loss
from .rl import compute_advantages, position_error, rollout_reward


PUFFER_REWARD_MODES = (
    "env",
    "progress",
    "progress_clearance",
    "progress_yaw_clearance",
    "live_clearance",
    "live_stable_clearance",
    "puffer_hover_transfer",
    "puffer_hover_transfer_strict",
    "puffer_drift_recovery",
    "puffer_drift_recovery_aggressive",
    "puffer_precontact_drift_brake",
    "puffer_precontact_transfer",
    "puffer_startup_drift_recovery",
)


@dataclass(frozen=True, slots=True)
class PufferPpoConfig:
    learning_rate: float = 3e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.002
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    update_epochs: int = 2
    minibatch_size: int = 4096
    action_std: float = 0.08
    imitation_coef: float = 0.0
    reference_coef: float = 0.2
    crash_replay_coef: float = 0.0
    crash_replay_batch_size: int = 512
    crash_replay_envelope_coef: float = 0.0
    crash_replay_action_abs_limit: float = 0.85
    transfer_replay_coef: float = 0.0
    transfer_replay_batch_size: int = 1024
    transfer_replay_envelope_coef: float = 0.0
    transfer_replay_action_abs_limit: float = 0.85
    previous_action_observation_scale: float = 1.0


def collect_puffer_rollout(
    env: SixDofCrazyflieEnv,
    policy: PufferSixDofPolicy,
    *,
    horizon: int,
    action_std: float,
    reward_mode: str,
    previous_action_observation_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    validate_puffer_env(env, policy)
    obs = env.observations.copy()
    observations = np.empty((horizon, env.num_envs, policy.metadata.observation_dim), dtype=np.float32)
    actions = np.empty((horizon, env.num_envs, ACTION_DIM), dtype=np.float32)
    teachers = np.empty_like(actions)
    log_probs = np.empty((horizon, env.num_envs), dtype=np.float32)
    rewards = np.empty((horizon, env.num_envs), dtype=np.float32)
    dones = np.empty((horizon, env.num_envs), dtype=np.float32)
    values = np.empty((horizon, env.num_envs), dtype=np.float32)
    task_indices = np.zeros(env.num_envs, dtype=np.int64)
    tasks = (env.task,)
    for step_idx in range(horizon):
        previous_error = position_error(env)
        previous_horizontal_speed = np.linalg.norm(env.velocity[:, :2], axis=1).astype(np.float32)
        previous_raw_action = obs[:, -ACTION_DIM:].copy()
        policy_obs = scale_previous_action_observation(obs, previous_action_observation_scale)
        with torch.no_grad():
            action, log_prob, _entropy, value = puffer_act(policy, torch.from_numpy(policy_obs).float(), action_std)
        action_np = action.cpu().numpy().astype(np.float32)
        teacher_np = teacher_actions(env, task=env.task).astype(np.float32)
        if hasattr(env, "set_native_context"):
            context_reward = "live_stable_clearance" if reward_mode.startswith("puffer_") else reward_mode
            env.set_native_context(task_indices=task_indices, tasks=tasks, reward_mode=context_reward, previous_error=previous_error)
        next_obs, base_reward, terminals, truncations, _ = env.step(action_np)
        done = terminals | truncations
        observations[step_idx] = policy_obs
        actions[step_idx] = action_np
        teachers[step_idx] = teacher_np
        log_probs[step_idx] = log_prob.cpu().numpy().astype(np.float32)
        values[step_idx] = value.cpu().numpy().astype(np.float32)
        rewards[step_idx] = puffer_rollout_reward(
            env,
            base_reward,
            done,
            previous_error,
            action_np,
            reward_mode,
            tasks,
            task_indices,
            previous_action=previous_raw_action,
            previous_horizontal_speed=previous_horizontal_speed,
        )
        dones[step_idx] = done.astype(np.float32)
        obs = env.reset_done(done).copy() if np.any(done) else next_obs.copy()
    with torch.no_grad():
        next_obs = scale_previous_action_observation(obs, previous_action_observation_scale)
        next_value = policy.value(torch.from_numpy(next_obs).float()).cpu().numpy().astype(np.float32)
    return {
        "observations": observations,
        "actions": actions,
        "teacher_actions": teachers,
        "log_probs": log_probs,
        "rewards": rewards,
        "dones": dones,
        "values": values,
        "next_value": next_value,
    }


def puffer_act(policy: PufferSixDofPolicy, observations: torch.Tensor, action_std: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = policy(observations)
    std = puffer_std(policy, action_std, observations.device)
    dist = torch.distributions.Normal(mean, std)
    raw_action = dist.rsample()
    action = raw_action.clamp(-1.0, 1.0)
    log_prob = dist.log_prob(action).sum(dim=1)
    entropy = dist.entropy().sum(dim=1)
    value = policy.value(observations)
    return action, log_prob, entropy, value


def puffer_evaluate_actions(policy: PufferSixDofPolicy, observations: torch.Tensor, actions: torch.Tensor, action_std: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = policy(observations)
    std = puffer_std(policy, action_std, observations.device)
    dist = torch.distributions.Normal(mean, std)
    clipped_actions = actions.clamp(-0.999, 0.999)
    return dist.log_prob(clipped_actions).sum(dim=1), dist.entropy().sum(dim=1), policy.value(observations)


def puffer_ppo_update(
    policy: PufferSixDofPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, np.ndarray],
    config: PufferPpoConfig,
    reference_policy: PufferSixDofPolicy | None = None,
    crash_replay: dict[str, torch.Tensor] | None = None,
    transfer_replay: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    observations = torch.from_numpy(rollout["observations"].reshape(-1, rollout["observations"].shape[-1])).float()
    actions = torch.from_numpy(rollout["actions"].reshape(-1, ACTION_DIM)).float()
    teacher = torch.from_numpy(rollout["teacher_actions"].reshape(-1, ACTION_DIM)).float()
    old_log_probs = torch.from_numpy(rollout["log_probs"].reshape(-1)).float()
    advantages_np, returns_np = compute_advantages(rollout, config.gamma, config.gae_lambda)
    advantages = torch.from_numpy((advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)).float()
    returns = torch.from_numpy(returns_np).float()
    losses = []
    order = torch.arange(observations.shape[0])
    for _ in range(config.update_epochs):
        order = order[torch.randperm(len(order))]
        for start in range(0, len(order), config.minibatch_size):
            idx = order[start : start + config.minibatch_size]
            log_prob, entropy, value = puffer_evaluate_actions(policy, observations[idx], actions[idx], config.action_std)
            ratio = (log_prob - old_log_probs[idx]).exp()
            policy_loss = -torch.min(ratio * advantages[idx], ratio.clamp(1.0 - config.clip_coef, 1.0 + config.clip_coef) * advantages[idx]).mean()
            value_loss = F.mse_loss(value, returns[idx])
            actor_output = policy(observations[idx])
            imitation_loss = F.mse_loss(actor_output, teacher[idx])
            reference_loss = puffer_reference_mse(reference_policy, actor_output, observations[idx])
            crash_replay_loss = puffer_crash_replay_mse(policy, crash_replay, config)
            transfer_replay_loss = puffer_transfer_replay_mse(policy, transfer_replay, config)
            loss = (
                policy_loss
                + config.value_coef * value_loss
                + config.imitation_coef * imitation_loss
                + config.reference_coef * reference_loss
                + config.crash_replay_coef * crash_replay_loss
                + config.transfer_replay_coef * transfer_replay_loss
                - config.entropy_coef * entropy.mean()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(
                (
                    float(policy_loss.detach()),
                    float(value_loss.detach()),
                    float(entropy.mean().detach()),
                    float(imitation_loss.detach()),
                    float(reference_loss.detach()),
                    float(crash_replay_loss.detach()),
                    float(transfer_replay_loss.detach()),
                )
            )
    return {
        "policy_loss": float(np.mean([item[0] for item in losses])),
        "value_loss": float(np.mean([item[1] for item in losses])),
        "entropy": float(np.mean([item[2] for item in losses])),
        "imitation_loss": float(np.mean([item[3] for item in losses])),
        "reference_loss": float(np.mean([item[4] for item in losses])),
        "crash_replay_loss": float(np.mean([item[5] for item in losses])),
        "transfer_replay_loss": float(np.mean([item[6] for item in losses])),
    }


def frozen_puffer_policy(policy: PufferSixDofPolicy) -> PufferSixDofPolicy:
    reference = copy.deepcopy(policy)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return reference


def puffer_reference_mse(reference_policy: PufferSixDofPolicy | None, actor_output: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
    if reference_policy is None:
        return actor_output.new_tensor(0.0)
    with torch.no_grad():
        target = reference_policy(observations)
    return F.mse_loss(actor_output, target)


def validate_puffer_env(env: SixDofCrazyflieEnv, policy: PufferSixDofPolicy) -> None:
    if policy.metadata.observation_dim != env.observations.shape[1]:
        raise ValueError(f"Puffer checkpoint expects obs_dim={policy.metadata.observation_dim}, env has {env.observations.shape[1]}")
    if policy.metadata.action_dim != ACTION_DIM:
        raise ValueError(f"Puffer checkpoint expects action_dim={policy.metadata.action_dim}, env has {ACTION_DIM}")


def puffer_std(policy: PufferSixDofPolicy, action_std: float, device: torch.device) -> torch.Tensor:
    return (torch.exp(policy.decoder.decoder_logstd).clamp(0.05, 2.0) * float(action_std)).to(device)


def puffer_rollout_reward(
    env: SixDofCrazyflieEnv,
    base_reward: np.ndarray,
    done: np.ndarray,
    previous_error: np.ndarray,
    actions: np.ndarray,
    mode: str,
    tasks: tuple[str, ...],
    task_indices: np.ndarray,
    previous_action: np.ndarray | None = None,
    previous_horizontal_speed: np.ndarray | None = None,
) -> np.ndarray:
    if mode not in PUFFER_REWARD_MODES:
        raise ValueError(f"unknown Puffer reward mode {mode!r}")
    if getattr(env, "use_native_step", False):
        return base_reward
    if mode == "puffer_drift_recovery":
        return drift_recovery_reward(env, done, previous_error, actions)
    if mode == "puffer_drift_recovery_aggressive":
        return aggressive_drift_recovery_reward(env, done, previous_error, actions)
    if mode == "puffer_precontact_transfer":
        return precontact_transfer_reward(env, done, previous_error, actions, previous_action=previous_action)
    if mode == "puffer_precontact_drift_brake":
        return precontact_drift_brake_reward(env, done, previous_error, actions, previous_action=previous_action)
    if mode == "puffer_startup_drift_recovery":
        return startup_drift_recovery_reward(
            env,
            done,
            previous_error,
            actions,
            previous_action=previous_action,
            previous_horizontal_speed=previous_horizontal_speed,
        )
    if not mode.startswith("puffer_hover_transfer"):
        return rollout_reward(env, base_reward, done, previous_error, actions, mode, tasks=tasks, task_indices=task_indices)
    return hover_transfer_reward(env, base_reward, done, previous_error, actions, mode, tasks, task_indices)
