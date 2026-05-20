from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import ACTION_DIM
from .policies import SixDofPolicy, teacher_actions


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


def collect_rollout(env, model: SixDofActorCritic, *, horizon: int, action_std: float) -> dict[str, np.ndarray]:
    observations, actions, log_probs, rewards, dones, values, teacher = [], [], [], [], [], [], []
    obs = env.observations.copy()
    for _ in range(horizon):
        obs_tensor = torch.from_numpy(obs).float()
        with torch.no_grad():
            action, log_prob, _entropy, value = model.act(obs_tensor, action_std)
        next_obs, reward, terminal, truncation, _info = env.step(action.cpu().numpy())
        done = terminal | truncation
        observations.append(obs.copy())
        actions.append(action.cpu().numpy().astype(np.float32))
        teacher.append(teacher_actions(env, task=env.task).copy())
        log_probs.append(log_prob.cpu().numpy().astype(np.float32))
        rewards.append(reward.copy())
        dones.append(done.astype(np.float32))
        values.append(value.cpu().numpy().astype(np.float32))
        obs = env.reset_done(done).copy() if np.any(done) else next_obs.copy()
    with torch.no_grad():
        next_value = model.critic(torch.from_numpy(obs).float()).squeeze(1).cpu().numpy().astype(np.float32)
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
