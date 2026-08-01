from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flightrl.bounded_action import BoundedNormal

from .controller import executed_action_for_controller, imitation_target_for_controller, validate_controller
from .env import ACTION_DIM
from .dataset import task_probability_vector, teacher_labels
from .episode_tasks import sample_task_indices
from .observation import augment_observation
from .policies import SixDofPolicy
from .ppo_reward import (
    position_error_for_task_indices,
    rollout_reward,
)
from .tasks import append_task_encoding


REWARD_MODES = ("env", "progress", "progress_clearance", "progress_yaw_clearance", "live_clearance", "live_stable_clearance")


@dataclass(frozen=True, slots=True)
class PpoConfig:
    hidden_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 4096
    action_std: float = 0.35
    imitation_coef: float = 0.0
    reference_coef: float = 0.0


class SixDofActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.actor = SixDofPolicy(hidden_size=hidden_size, input_dim=input_dim)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(ACTION_DIM))

    def act(self, observations: torch.Tensor, action_std: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        location = self.actor.action_location(observations)
        std = torch.exp(self.log_std).clamp(0.05, 2.0) * action_std
        dist = BoundedNormal(location, std)
        action, pre_tanh = dist.rsample_with_pre_tanh()
        log_prob = dist.log_prob_from_pre_tanh(pre_tanh)
        entropy = dist.entropy()
        value = self.critic(observations).squeeze(1)
        return action, pre_tanh, log_prob, entropy, value

    def evaluate_actions(self, observations: torch.Tensor, pre_tanh_actions: torch.Tensor, action_std: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        location = self.actor.action_location(observations)
        std = torch.exp(self.log_std).clamp(0.05, 2.0) * action_std
        dist = BoundedNormal(location, std)
        log_prob = dist.log_prob_from_pre_tanh(pre_tanh_actions)
        entropy = dist.entropy()
        value = self.critic(observations).squeeze(1)
        return log_prob, entropy, value


def load_actor_checkpoint(model: SixDofActorCritic, checkpoint: dict | None) -> None:
    if checkpoint is not None:
        model.actor.load_state_dict(checkpoint["state_dict"])


def collect_rollout(
    env,
    model: SixDofActorCritic,
    *,
    horizon: int,
    action_std: float,
    reward_mode: str = "env",
    observation_mode: str = "base",
    tasks: tuple[str, ...] | None = None,
    rng: np.random.Generator | None = None,
    task_probabilities: np.ndarray | None = None,
    controller: str = "policy",
    residual_scale: float = 0.0,
) -> dict[str, np.ndarray]:
    validate_controller(controller)
    obs = env.observations.copy()
    previous_obs = None
    previous_action = np.zeros((env.num_envs, ACTION_DIM), dtype=np.float32)
    fresh = np.ones(env.num_envs, dtype=bool)
    tasks = tasks or (env.task,)
    rng = rng or np.random.default_rng(0)
    task_probabilities = task_probabilities if task_probabilities is not None else task_probability_vector(tasks)
    task_indices = sample_task_indices(
        rng,
        env.num_envs,
        tasks,
        task_probabilities,
    )
    if hasattr(env, "set_native_context"):
        env.set_native_context(
            task_indices=task_indices,
            tasks=tasks,
            reward_mode=reward_mode,
        )
        obs = env.observations.copy()
    first_model_obs = append_task_encoding(obs, task_indices, len(tasks))
    first_policy_obs = augment_observation(first_model_obs, first_model_obs, previous_action, observation_mode)
    observations = np.empty((horizon, env.num_envs, first_policy_obs.shape[1]), dtype=np.float32)
    actions = np.empty((horizon, env.num_envs, ACTION_DIM), dtype=np.float32)
    pre_tanh_actions = np.empty_like(actions)
    executed_actions = np.empty_like(actions)
    teacher = np.empty_like(actions)
    log_probs = np.empty((horizon, env.num_envs), dtype=np.float32)
    rewards = np.empty((horizon, env.num_envs), dtype=np.float32)
    dones = np.empty((horizon, env.num_envs), dtype=np.float32)
    values = np.empty((horizon, env.num_envs), dtype=np.float32)
    for step_idx in range(horizon):
        model_obs = append_task_encoding(obs.copy(), task_indices, len(tasks))
        if previous_obs is None:
            previous_obs = model_obs.copy()
        previous_obs[fresh] = model_obs[fresh]
        policy_obs = augment_observation(model_obs, previous_obs, previous_action, observation_mode)
        obs_tensor = torch.from_numpy(policy_obs).float()
        with torch.no_grad():
            action, pre_tanh, log_prob, _entropy, value = model.act(obs_tensor, action_std)
        teacher_action = teacher_labels(env, tasks, task_indices).copy()
        previous_error = position_error_for_task_indices(env, tasks, task_indices)
        action_np = action.cpu().numpy()
        executed_action = executed_action_for_controller(controller, action_np, teacher_action, residual_scale)
        if hasattr(env, "set_native_context"):
            env.set_native_context(task_indices=task_indices, tasks=tasks, reward_mode=reward_mode, previous_error=previous_error)
        next_obs, reward, terminal, truncation, _info = env.step(executed_action)
        done = terminal | truncation
        observations[step_idx] = policy_obs
        actions[step_idx] = action_np.astype(np.float32)
        pre_tanh_actions[step_idx] = pre_tanh.cpu().numpy().astype(np.float32)
        executed_actions[step_idx] = executed_action.astype(np.float32)
        teacher[step_idx] = imitation_target_for_controller(controller, teacher_action)
        log_probs[step_idx] = log_prob.cpu().numpy().astype(np.float32)
        rewards[step_idx] = reward if getattr(env, "use_native_step", False) else rollout_reward(env, reward, done, previous_error, executed_action, reward_mode, tasks=tasks, task_indices=task_indices)
        dones[step_idx] = done.astype(np.float32)
        values[step_idx] = value.cpu().numpy().astype(np.float32)
        previous_obs = model_obs.copy()
        previous_action = executed_action.astype(np.float32)
        fresh[:] = False
        obs = env.reset_done(done).copy() if np.any(done) else next_obs.copy()
        if np.any(done):
            reset_mask = done.astype(bool)
            previous_action[reset_mask] = 0.0
            fresh = reset_mask
            task_indices[reset_mask] = sample_task_indices(
                rng,
                int(np.sum(reset_mask)),
                tasks,
                task_probabilities,
            )
            if hasattr(env, "set_native_context"):
                env.set_native_context(
                    task_indices=task_indices,
                    tasks=tasks,
                    reward_mode=reward_mode,
                )
                obs = env.observations.copy()
    with torch.no_grad():
        model_obs = append_task_encoding(obs.copy(), task_indices, len(tasks))
        previous_obs[fresh] = model_obs[fresh]
        next_obs = augment_observation(model_obs, previous_obs, previous_action, observation_mode)
        next_value = model.critic(torch.from_numpy(next_obs).float()).squeeze(1).cpu().numpy().astype(np.float32)
    return {
        "observations": observations,
        "actions": actions,
        "pre_tanh_actions": pre_tanh_actions,
        "executed_actions": executed_actions,
        "teacher_actions": teacher,
        "log_probs": log_probs,
        "rewards": rewards,
        "dones": dones,
        "values": values,
        "next_value": next_value,
    }


def compute_advantages(rollout: dict[str, np.ndarray], gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    next_value = rollout["next_value"]
    for step in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * next_nonterminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
        advantages[step] = last_advantage
        next_value = values[step]
    returns = advantages + values
    return advantages.reshape(-1), returns.reshape(-1)


def ppo_update(model: SixDofActorCritic, optimizer, rollout: dict[str, np.ndarray], config: PpoConfig, reference_actor: SixDofPolicy | None = None) -> dict[str, float]:
    observations = torch.from_numpy(rollout["observations"].reshape(-1, rollout["observations"].shape[-1])).float()
    pre_tanh_actions = torch.from_numpy(rollout["pre_tanh_actions"].reshape(-1, ACTION_DIM)).float()
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
            log_prob, entropy, value = model.evaluate_actions(observations[idx], pre_tanh_actions[idx], config.action_std)
            ratio = (log_prob - old_log_probs[idx]).exp()
            policy_loss = -torch.min(ratio * advantages[idx], ratio.clamp(1.0 - config.clip_coef, 1.0 + config.clip_coef) * advantages[idx]).mean()
            value_loss = F.mse_loss(value, returns[idx])
            actor_output = model.actor(observations[idx])
            imitation_loss = F.mse_loss(actor_output, teacher[idx])
            reference_loss = reference_mse(reference_actor, actor_output, observations[idx])
            loss = (
                policy_loss
                + config.value_coef * value_loss
                + config.imitation_coef * imitation_loss
                + config.reference_coef * reference_loss
                - config.entropy_coef * entropy.mean()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(
                (
                    float(policy_loss.detach()),
                    float(value_loss.detach()),
                    float(entropy.mean().detach()),
                    float(imitation_loss.detach()),
                    float(reference_loss.detach()),
                )
            )
    return {
        "policy_loss": float(np.mean([item[0] for item in losses])),
        "value_loss": float(np.mean([item[1] for item in losses])),
        "entropy": float(np.mean([item[2] for item in losses])),
        "imitation_loss": float(np.mean([item[3] for item in losses])),
        "reference_loss": float(np.mean([item[4] for item in losses])),
    }


def reference_mse(reference_actor: SixDofPolicy | None, actor_output: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
    if reference_actor is None:
        return actor_output.new_tensor(0.0)
    with torch.no_grad():
        target = reference_actor(observations)
    return F.mse_loss(actor_output, target)
