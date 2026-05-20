from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import ACTION_DIM
from .observation import augment_observation
from .policies import SixDofPolicy, teacher_actions


REWARD_MODES = ("env", "progress", "progress_clearance")


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

    def act(self, observations: torch.Tensor, action_std: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(observations)
        std = torch.exp(self.log_std).clamp(0.05, 2.0) * action_std
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = raw_action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        value = self.critic(observations).squeeze(1)
        return action, log_prob, entropy, value

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor, action_std: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(observations)
        std = torch.exp(self.log_std).clamp(0.05, 2.0) * action_std
        dist = torch.distributions.Normal(mean, std)
        clipped_actions = actions.clamp(-0.999, 0.999)
        log_prob = dist.log_prob(clipped_actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
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
) -> dict[str, np.ndarray]:
    observations, actions, log_probs, rewards, dones, values, teacher = [], [], [], [], [], [], []
    obs = env.observations.copy()
    previous_obs = None
    previous_action = np.zeros((env.num_envs, ACTION_DIM), dtype=np.float32)
    fresh = np.ones(env.num_envs, dtype=bool)
    for _ in range(horizon):
        model_obs = obs.copy()
        if previous_obs is None:
            previous_obs = model_obs.copy()
        previous_obs[fresh] = model_obs[fresh]
        policy_obs = augment_observation(model_obs, previous_obs, previous_action, observation_mode)
        obs_tensor = torch.from_numpy(policy_obs).float()
        with torch.no_grad():
            action, log_prob, _entropy, value = model.act(obs_tensor, action_std)
        teacher_action = teacher_actions(env, task=env.task).copy()
        previous_error = position_error(env)
        action_np = action.cpu().numpy()
        next_obs, reward, terminal, truncation, _info = env.step(action_np)
        done = terminal | truncation
        observations.append(policy_obs.copy())
        actions.append(action_np.astype(np.float32))
        teacher.append(teacher_action)
        log_probs.append(log_prob.cpu().numpy().astype(np.float32))
        rewards.append(rollout_reward(env, reward, done, previous_error, action_np, reward_mode))
        dones.append(done.astype(np.float32))
        values.append(value.cpu().numpy().astype(np.float32))
        previous_obs = model_obs.copy()
        previous_action = action_np.astype(np.float32)
        fresh[:] = False
        obs = env.reset_done(done).copy() if np.any(done) else next_obs.copy()
        if np.any(done):
            previous_action[done.astype(bool)] = 0.0
            fresh = done.astype(bool)
    with torch.no_grad():
        previous_obs[fresh] = obs[fresh]
        next_obs = augment_observation(obs, previous_obs, previous_action, observation_mode)
        next_value = model.critic(torch.from_numpy(next_obs).float()).squeeze(1).cpu().numpy().astype(np.float32)
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "teacher_actions": np.asarray(teacher, dtype=np.float32),
        "log_probs": np.asarray(log_probs, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "next_value": next_value,
    }


def position_error(env) -> np.ndarray:
    return np.linalg.norm(env.target_position - env.position, axis=1).astype(np.float32)


def rollout_reward(env, base_reward: np.ndarray, done: np.ndarray, previous_error: np.ndarray, actions: np.ndarray, mode: str) -> np.ndarray:
    if mode == "env":
        return base_reward.copy()
    if mode == "progress":
        return shaped_progress_reward(env, done, previous_error, actions, clearance_threshold=0.25, clearance_weight=1.0)
    if mode == "progress_clearance":
        return shaped_progress_reward(env, done, previous_error, actions, clearance_threshold=0.45, clearance_weight=2.5)
    raise ValueError(f"unknown PPO reward mode {mode!r}")


def shaped_progress_reward(
    env,
    done: np.ndarray,
    previous_error: np.ndarray,
    actions: np.ndarray,
    *,
    clearance_threshold: float,
    clearance_weight: float,
) -> np.ndarray:
    current_error = position_error(env)
    progress = previous_error - current_error
    speed = np.linalg.norm(env.velocity, axis=1)
    yaw_error = np.abs(env.observations[:, 16])
    clearance_penalty = np.maximum(0.0, clearance_threshold - np.min(env.ranges_m[:, :4], axis=1))
    control = np.linalg.norm(actions, axis=1)
    reward = 0.2 + 3.0 * progress - 0.05 * current_error - 0.02 * speed - 0.1 * yaw_error - clearance_weight * clearance_penalty - 0.01 * control
    reward -= done.astype(np.float32)
    return reward.astype(np.float32)


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
            log_prob, entropy, value = model.evaluate_actions(observations[idx], actions[idx], config.action_std)
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
