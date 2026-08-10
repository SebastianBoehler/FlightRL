from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .range_policy import RangeExplorationActorCritic


@dataclass(frozen=True, slots=True)
class RangePpoConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    update_epochs: int = 4
    action_std: float = 0.25
    frontier_aux_coef: float = 0.0
    shield_aux_coef: float = 0.10
    turn_commitment_coef: float = 0.0


def collect_range_rollout(
    env,
    model: RangeExplorationActorCritic,
    *,
    horizon: int,
    action_std: float,
    reset_seed: int,
) -> dict[str, np.ndarray]:
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("range PPO horizon must be positive")
    observations = np.empty((horizon, env.num_envs, 4106), dtype=np.float32)
    actions = np.empty((horizon, env.num_envs, 2), dtype=np.float32)
    executed_actions = np.empty_like(actions)
    shield_overrides = np.empty((horizon, env.num_envs), dtype=bool)
    pre_tanh = np.empty_like(actions)
    log_probs = np.empty((horizon, env.num_envs), dtype=np.float32)
    rewards = np.empty_like(log_probs)
    dones = np.empty_like(log_probs)
    values = np.empty_like(log_probs)
    obs = env.observations.copy()
    for step in range(horizon):
        obs_tensor = torch.from_numpy(obs)
        with torch.no_grad():
            distribution, value = model.distribution(obs_tensor, action_std)
            action, raw = distribution.sample_with_pre_tanh()
            log_prob = distribution.log_prob_from_pre_tanh(raw)
        action_np = action.cpu().numpy().astype(np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated | truncated
        observations[step] = obs
        actions[step] = action_np
        executed_actions[step] = np.stack(
            [item.previous_action for item in env.envs]
        )
        shield_overrides[step] = np.asarray(
            [
                row["forward_clearance_override"] or row["safety_terminal"]
                for row in info
            ],
            dtype=bool,
        )
        pre_tanh[step] = raw.cpu().numpy().astype(np.float32)
        log_probs[step] = log_prob.cpu().numpy().astype(np.float32)
        rewards[step] = reward
        dones[step] = done.astype(np.float32)
        values[step] = value.cpu().numpy().astype(np.float32)
        obs = (
            env.reset_done(done, seed=reset_seed + step * env.num_envs)
            if np.any(done)
            else next_obs
        )
    with torch.no_grad():
        _dist, next_value = model.distribution(torch.from_numpy(obs), action_std)
    return {
        "observations": observations,
        "actions": actions,
        "executed_actions": executed_actions,
        "shield_overrides": shield_overrides,
        "pre_tanh_actions": pre_tanh,
        "log_probs": log_probs,
        "rewards": rewards,
        "dones": dones,
        "values": values,
        "next_value": next_value.cpu().numpy().astype(np.float32),
        "actor_observation_contains_truth": np.asarray(0, dtype=np.uint8),
    }


def range_ppo_update(
    model: RangeExplorationActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, np.ndarray],
    config: RangePpoConfig,
) -> dict[str, float]:
    advantages, returns = _advantages(
        rollout, gamma=config.gamma, gae_lambda=config.gae_lambda
    )
    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    observations = torch.from_numpy(rollout["observations"])
    raw_actions = torch.from_numpy(rollout["pre_tanh_actions"])
    old_log_probs = torch.from_numpy(rollout["log_probs"])
    advantage_tensor = torch.from_numpy(normalized)
    return_tensor = torch.from_numpy(returns)
    executed_actions = torch.from_numpy(rollout["executed_actions"])
    shield_overrides = torch.from_numpy(rollout["shield_overrides"])
    losses: list[tuple[float, float, float, float, float, float]] = []
    for _ in range(config.update_epochs):
        log_prob_steps = []
        entropy_steps = []
        value_steps = []
        mode_steps = []
        frontier_loss_steps = []
        turn_commitment_loss_steps = []
        for step in range(observations.shape[0]):
            distribution, value = model.distribution(
                observations[step], config.action_std
            )
            log_prob_steps.append(
                distribution.log_prob_from_pre_tanh(raw_actions[step])
            )
            entropy_steps.append(distribution.entropy())
            value_steps.append(value)
            mode_steps.append(distribution.mode)
            target_yaw, frontier_active = frontier_yaw_targets(observations[step])
            if bool(frontier_active.any()):
                frontier_loss_steps.append(
                    F.mse_loss(
                        distribution.mode[frontier_active, 1],
                        target_yaw[frontier_active],
                    )
                )
            committed_yaw, turn_active = turn_commitment_targets(
                observations[step], shield_overrides[step]
            )
            if bool(turn_active.any()):
                turn_commitment_loss_steps.append(
                    F.mse_loss(
                        distribution.mode[turn_active, 1],
                        committed_yaw[turn_active],
                    )
                )
        log_prob = torch.stack(log_prob_steps)
        entropy = torch.stack(entropy_steps)
        value = torch.stack(value_steps)
        modes = torch.stack(mode_steps)
        ratio = (log_prob - old_log_probs).exp()
        policy_loss = -torch.minimum(
            ratio * advantage_tensor,
            ratio.clamp(1.0 - config.clip_coef, 1.0 + config.clip_coef)
            * advantage_tensor,
        ).mean()
        value_loss = F.mse_loss(value, return_tensor)
        frontier_loss = (
            torch.stack(frontier_loss_steps).mean()
            if frontier_loss_steps
            else value.new_tensor(0.0)
        )
        shield_loss = (
            F.mse_loss(modes[shield_overrides], executed_actions[shield_overrides])
            if bool(shield_overrides.any())
            else value.new_tensor(0.0)
        )
        turn_commitment_loss = (
            torch.stack(turn_commitment_loss_steps).mean()
            if turn_commitment_loss_steps
            else value.new_tensor(0.0)
        )
        loss = (
            policy_loss
            + config.value_coef * value_loss
            - config.entropy_coef * entropy.mean()
            + config.frontier_aux_coef * frontier_loss
            + config.shield_aux_coef * shield_loss
            + config.turn_commitment_coef * turn_commitment_loss
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
                float(frontier_loss.detach()),
                float(shield_loss.detach()),
                float(turn_commitment_loss.detach()),
            )
        )
    return {
        "policy_loss": float(np.mean([value[0] for value in losses])),
        "value_loss": float(np.mean([value[1] for value in losses])),
        "entropy": float(np.mean([value[2] for value in losses])),
        "frontier_aux_loss": float(np.mean([value[3] for value in losses])),
        "shield_aux_loss": float(np.mean([value[4] for value in losses])),
        "turn_commitment_loss": float(np.mean([value[5] for value in losses])),
    }


def frontier_yaw_targets(
    observations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if observations.ndim != 2 or observations.shape[1] != 4106:
        raise ValueError("frontier yaw targets require batched range observations")
    frontier = observations[:, :4096].reshape(-1, 4, 32, 32)[:, 3]
    mass = frontier.sum(dim=(1, 2))
    coordinates = torch.arange(32, dtype=observations.dtype, device=observations.device)
    rows = coordinates[None, :, None]
    columns = coordinates[None, None, :]
    denominator = mass.clamp_min(1.0)
    forward = ((16.0 - rows) * frontier).sum(dim=(1, 2)) / denominator
    left = ((16.0 - columns) * frontier).sum(dim=(1, 2)) / denominator
    targets = torch.atan2(left, forward.clamp_min(0.5)) / (torch.pi / 2.0)
    return targets.clamp(-1.0, 1.0), mass > 0.0


def turn_commitment_targets(
    observations: torch.Tensor,
    shield_overrides: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if observations.ndim != 2 or observations.shape[1] != 4106:
        raise ValueError("turn commitment requires batched range observations")
    blocked = torch.as_tensor(
        shield_overrides,
        dtype=torch.bool,
        device=observations.device,
    )
    if blocked.shape != (observations.shape[0],):
        raise ValueError("turn commitment shield flags do not match observations")
    previous_yaw = observations[:, 4105]
    return previous_yaw, blocked & (previous_yaw.abs() >= 0.10)


def _advantages(
    rollout: dict[str, np.ndarray],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last = np.zeros(rewards.shape[1], dtype=np.float32)
    next_value = rollout["next_value"]
    for step in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        last = delta + gamma * gae_lambda * nonterminal * last
        advantages[step] = last
        next_value = values[step]
    return advantages, advantages + values
