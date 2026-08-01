from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from flightrl.puffer4_door_advantage import (
    DoorRollout,
    generalized_advantage,
)
from flightrl.puffer4_door_imitation import _forward_sequence
from flightrl.puffer4_door_policy import (
    DOOR_PIXELS,
    DOOR_POLICY_OBS_DIM,
    DOOR_PRIVILEGED_DIM,
)
from flightrl.puffer4_door_observation import DOOR_PROPRIO_DIM


DOOR_IMAGE_DIM = 3 * DOOR_PIXELS
DOOR_CRITIC_INPUT_DIM = DOOR_PROPRIO_DIM + DOOR_PRIVILEGED_DIM


def privileged_door_features(observations: torch.Tensor) -> torch.Tensor:
    proprio = observations[..., DOOR_IMAGE_DIM:DOOR_POLICY_OBS_DIM]
    privileged = observations[..., DOOR_POLICY_OBS_DIM:]
    features = torch.cat((proprio, privileged), dim=-1).float()
    if features.shape[-1] != DOOR_CRITIC_INPUT_DIM:
        raise ValueError("asymmetric critic received the wrong door contract")
    return features


class DoorAsymmetricCritic(nn.Module):
    input_dim = DOOR_CRITIC_INPUT_DIM

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(privileged_door_features(observations)).squeeze(-1)


def actor_parameters(policy) -> tuple[nn.Parameter, ...]:
    excluded = {
        id(parameter)
        for parameter in policy.decoder.value_function.parameters()
    }
    excluded.add(id(policy.decoder.decoder_logstd))
    return tuple(
        parameter
        for parameter in policy.parameters()
        if parameter.requires_grad and id(parameter) not in excluded
    )


@dataclass(frozen=True, slots=True)
class DoorAsymmetricConfig:
    horizon: int = 64
    learning_rate: float = 1.0e-4
    critic_learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coefficient: float = 0.10
    entropy_coefficient: float = 0.001
    value_coefficient: float = 0.5
    imitation_coefficient: float = 0.25
    policy_logstd: float = -3.0
    max_grad_norm: float = 1.0
    optimization_epochs: int = 2
    minibatch_agents: int = 32


class DoorAsymmetricTrainer:
    def __init__(self, policy, vec, torch_pufferl, config: DoorAsymmetricConfig):
        self.policy = policy
        self.vec = vec
        self.config = config
        self.observations = torch_pufferl._cpu_tensor(
            vec.obs_ptr,
            (vec.total_agents, vec.obs_size),
            torch.float32,
        )
        self.rewards = torch_pufferl._cpu_tensor(
            vec.rewards_ptr,
            (vec.total_agents,),
            torch.float32,
        )
        self.terminals = torch_pufferl._cpu_tensor(
            vec.terminals_ptr,
            (vec.total_agents,),
            torch.float32,
        )
        self.critic = DoorAsymmetricCritic()
        with torch.no_grad():
            policy.decoder.decoder_logstd.fill_(config.policy_logstd)
        parameters = [
            {"params": actor_parameters(policy), "lr": config.learning_rate},
            {
                "params": self.critic.parameters(),
                "lr": config.critic_learning_rate,
            },
        ]
        self.optimizer = torch.optim.AdamW(parameters)
        self.state = policy.initial_state(vec.total_agents, device="cpu")
        self.vec.reset()
    @torch.no_grad()
    def collect(self) -> DoorRollout:
        initial_state = tuple(item.clone() for item in self.state)
        buffers: dict[str, list[torch.Tensor]] = {
            key: []
            for key in (
                "observations",
                "actions",
                "log_probabilities",
                "rewards",
                "terminals",
                "values",
            )
        }
        for _ in range(self.config.horizon):
            observation = self.observations.clone()
            distribution, _, next_state = self.policy.forward_eval(
                observation,
                self.state,
            )
            action = distribution.sample()
            action[:, 0].clamp_(0.0, 1.0)
            action[:, 1].clamp_(-1.0, 1.0)
            log_probability = distribution.log_prob(action).sum(-1)
            value = self.critic(observation)
            self.vec.cpu_step(action.contiguous().data_ptr())
            terminal = self.terminals.clone()
            alive = (1.0 - terminal).view(1, -1, 1)
            self.state = tuple(item * alive for item in next_state)
            buffers["observations"].append(observation)
            buffers["actions"].append(action)
            buffers["log_probabilities"].append(log_probability)
            buffers["rewards"].append(self.rewards.clone())
            buffers["terminals"].append(terminal)
            buffers["values"].append(value)
        bootstrap = self.critic(self.observations).detach()
        stacked = {
            key: torch.stack(values)
            for key, values in buffers.items()
        }
        advantages, returns = generalized_advantage(
            stacked["rewards"],
            stacked["values"],
            stacked["terminals"],
            bootstrap,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        return DoorRollout(
            **stacked,
            advantages=advantages,
            returns=returns,
            initial_state=initial_state,
        )
    def optimize(self, rollout: DoorRollout) -> dict[str, float]:
        config = self.config
        observations = rollout.observations.transpose(0, 1).contiguous()
        actions = rollout.actions.transpose(0, 1).contiguous()
        old_log_probabilities = (
            rollout.log_probabilities.transpose(0, 1).contiguous()
        )
        old_values = rollout.values.transpose(0, 1).contiguous()
        advantages = rollout.advantages.transpose(0, 1).contiguous()
        returns = rollout.returns.transpose(0, 1).contiguous()
        terminals = rollout.terminals.transpose(0, 1).contiguous()
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1.0e-8
        )
        totals: dict[str, float] = {}
        batches = 0
        for _ in range(config.optimization_epochs):
            permutation = torch.randperm(self.vec.total_agents)
            for start in range(0, self.vec.total_agents, config.minibatch_agents):
                indices = permutation[start : start + config.minibatch_agents]
                state = tuple(item[:, indices].clone() for item in rollout.initial_state)
                distribution, _, _ = _forward_sequence(
                    self.policy,
                    observations[indices],
                    terminals[indices],
                    state,
                )
                flat_actions = actions[indices].reshape(-1, 2)
                new_log_probability = distribution.log_prob(flat_actions).sum(-1)
                old_log_probability = old_log_probabilities[indices].reshape(-1)
                ratio = torch.exp(new_log_probability - old_log_probability)
                advantage = advantages[indices].reshape(-1)
                policy_loss = torch.maximum(
                    -advantage * ratio,
                    -advantage
                    * ratio.clamp(
                        1.0 - config.clip_coefficient,
                        1.0 + config.clip_coefficient,
                    ),
                ).mean()
                batch_observations = observations[indices]
                value = self.critic(batch_observations).reshape(-1)
                old_value = old_values[indices].reshape(-1)
                target_return = returns[indices].reshape(-1)
                clipped_value = old_value + (value - old_value).clamp(
                    -config.clip_coefficient,
                    config.clip_coefficient,
                )
                value_loss = 0.5 * torch.maximum(
                    (value - target_return) ** 2,
                    (clipped_value - target_return) ** 2,
                ).mean()
                teacher = batch_observations[
                    ..., DOOR_POLICY_OBS_DIM : DOOR_POLICY_OBS_DIM + 2
                ].reshape(-1, 2)
                imitation_loss = (
                    (distribution.mean - teacher) ** 2
                    * teacher.new_tensor((1.0, 3.0))
                ).mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    + config.imitation_coefficient * imitation_loss
                    - config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                parameters = (
                    *actor_parameters(self.policy),
                    *tuple(self.critic.parameters()),
                )
                torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
                self.optimizer.step()
                metrics = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "imitation_loss": imitation_loss,
                    "entropy": entropy,
                    "approx_kl": (
                        (ratio - 1.0) - torch.log(ratio)
                    ).mean(),
                }
                for key, value_item in metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value_item.detach())
                batches += 1
        return {key: value / batches for key, value in totals.items()}
    def train(self, rollouts: int, *, log_interval: int = 4) -> list[dict]:
        history = []
        started = perf_counter()
        steps = 0
        for update in range(1, rollouts + 1):
            rollout = self.collect()
            losses = self.optimize(rollout)
            steps += self.vec.total_agents * self.config.horizon
            if update == 1 or update == rollouts or update % log_interval == 0:
                elapsed = perf_counter() - started
                row = {
                    "update": update,
                    "steps": steps,
                    "sps": steps / elapsed,
                    "loss": losses,
                    "env": dict(self.vec.log()),
                }
                history.append(row)
                print(
                    f"asymmetric_ppo={update}/{rollouts} steps={steps} "
                    f"sps={row['sps']:.0f} env={row['env']} loss={losses}",
                    flush=True,
                )
        return history
