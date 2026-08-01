from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


class StatefulSemanticPuffeRL:
    def __init__(self, trainer, torch_pufferl) -> None:
        self._trainer = trainer
        self._module = torch_pufferl
        horizon = int(trainer.config["horizon"])
        self.state_resets = torch.zeros(
            horizon,
            trainer.total_agents,
            device=trainer.device,
        )
        self.previous_terminals = torch.zeros(
            trainer.total_agents,
            device=trainer.device,
        )
        self.rollout_initial_state = _clone_state(trainer.state)

    def __getattr__(self, name):
        return getattr(self._trainer, name)

    def rollouts(self) -> None:
        trainer = self._trainer
        module = self._module
        profile = trainer.profile
        profile_type = module.Profile
        horizon = int(trainer.config["horizon"])
        state = _mask_state(trainer.state, self.previous_terminals)
        self.rollout_initial_state = _clone_state(state)
        self.state_resets.zero_()

        observations = trainer.vec_obs
        terminals = self.previous_terminals
        profile.mark(0)
        for step in range(horizon):
            if step > 0:
                self.state_resets[step] = terminals
                state = _mask_state(state, terminals)
            device_observations = torch.as_tensor(
                observations,
                device=trainer.device,
            )
            profile.mark(1)
            with torch.no_grad():
                distribution, value, state = trainer.policy.forward_eval(
                    device_observations,
                    state,
                )
                action, logprob, _ = module.sample_logits(distribution)
            profile.mark(2)

            trainer.observations[step] = device_observations
            trainer.actions[step] = action
            trainer.logprobs[step] = logprob
            trainer.values[step] = value.flatten()

            flat_action = (
                action.to(dtype=torch.float32)
                .reshape(trainer.total_agents, -1)
                .contiguous()
            )
            if trainer.gpu:
                flat_action = flat_action.cuda()
                trainer._vec.gpu_step(flat_action.data_ptr())
                torch.cuda.synchronize()
            else:
                trainer._vec.cpu_step(flat_action.data_ptr())
            observations = trainer.vec_obs
            rewards = torch.as_tensor(
                trainer.vec_rewards,
                device=trainer.device,
            )
            terminals = torch.as_tensor(
                trainer.vec_terminals,
                device=trainer.device,
            ).float()
            trainer.rewards[step] = rewards
            trainer.terminals[step] = terminals

            profile.mark(3)
            profile.elapsed(profile_type.EVAL_GPU, 1, 2)
            profile.elapsed(profile_type.EVAL_ENV, 2, 3)

        trainer.state = _clone_state(state)
        self.previous_terminals = terminals.detach().clone()
        profile.mark(1)
        profile.elapsed(profile_type.ROLLOUT, 0, 1)
        trainer.global_step += trainer.total_agents * horizon
        trainer.env_logs = trainer._vec.log()

    def train(self) -> None:
        trainer = self._trainer
        module = self._module
        config = trainer.config
        losses = defaultdict(float)

        beta0 = config["prio_beta0"]
        alpha = config["prio_alpha"]
        clip_coefficient = config["clip_coef"]
        value_clip = config["vf_clip_coef"]
        annealed_beta = (
            beta0
            + (1.0 - beta0)
            * alpha
            * trainer.epoch
            / trainer.total_epochs
        )
        trainer.ratio[:] = 1
        _anneal_learning_rate(trainer)

        observations = trainer.observations.transpose(0, 1).contiguous()
        actions = trainer.actions.transpose(0, 1).contiguous()
        values = trainer.values.T.contiguous()
        logprobs = trainer.logprobs.T.contiguous()
        rewards = trainer.rewards.T.contiguous().clamp(-1, 1)
        terminals = trainer.terminals.T.contiguous()
        state_resets = self.state_resets.T.contiguous()

        profile = trainer.profile
        profile_type = module.Profile
        profile.mark(0)
        minibatch_count = int(
            config["replay_ratio"]
            * trainer.batch_size
            / config["minibatch_size"]
        )
        for _ in range(minibatch_count):
            advantages = module.compute_puff_advantage(
                values,
                rewards,
                terminals,
                trainer.ratio,
                torch.zeros_like(values),
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )
            priorities = torch.nan_to_num(
                advantages.abs().sum(axis=1) ** alpha,
                0,
                0,
                0,
            )
            probabilities = (priorities + 1e-6) / (
                priorities.sum() + 1e-6
            )
            indices = torch.multinomial(
                probabilities,
                trainer.minibatch_segments,
                replacement=True,
            )
            priority = (
                trainer.total_agents * probabilities[indices, None]
            ) ** -annealed_beta
            initial_state = _slice_state(self.rollout_initial_state, indices)

            minibatch_observations = observations[indices]
            minibatch_actions = actions[indices]
            minibatch_logprobs = logprobs[indices]
            minibatch_values = values[indices]
            minibatch_returns = advantages[indices] + minibatch_values
            minibatch_advantages = advantages[indices]

            profile.mark(1)
            distribution, new_values, _ = (
                trainer.policy.forward_train_recurrent(
                    minibatch_observations,
                    initial_state,
                    state_resets[indices],
                )
            )
            _, new_logprobs, entropy = module.sample_logits(
                distribution,
                action=minibatch_actions,
            )
            profile.mark(2)
            profile.elapsed(profile_type.TRAIN_FORWARD, 1, 2)

            new_logprobs = new_logprobs.reshape(minibatch_logprobs.shape)
            log_ratio = new_logprobs - minibatch_logprobs
            ratio = log_ratio.exp()
            trainer.ratio[indices] = ratio.detach()
            with torch.no_grad():
                old_approximate_kl = (-log_ratio).mean()
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > clip_coefficient
                ).float().mean()

            normalized_advantages = priority * (
                minibatch_advantages - minibatch_advantages.mean()
            ) / (minibatch_advantages.std() + 1e-8)
            policy_loss = torch.max(
                -normalized_advantages * ratio,
                -normalized_advantages
                * torch.clamp(
                    ratio,
                    1.0 - clip_coefficient,
                    1.0 + clip_coefficient,
                ),
            ).mean()

            new_values = new_values.view(minibatch_returns.shape)
            clipped_values = minibatch_values + torch.clamp(
                new_values - minibatch_values,
                -value_clip,
                value_clip,
            )
            value_loss = 0.5 * torch.max(
                (new_values - minibatch_returns) ** 2,
                (clipped_values - minibatch_returns) ** 2,
            ).mean()
            entropy_loss = entropy.mean()
            loss = (
                policy_loss
                + config["vf_coef"] * value_loss
                - config["ent_coef"] * entropy_loss
            )
            values[indices] = new_values.detach().float()
            _record_losses(
                losses,
                policy_loss,
                value_loss,
                entropy_loss,
                old_approximate_kl,
                approximate_kl,
                clip_fraction,
                ratio,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainer.policy.parameters(),
                config["max_grad_norm"],
            )
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

        profile.mark(1)
        profile.elapsed(profile_type.TRAIN, 0, 1)
        trainer.losses = {
            key: value.item() / minibatch_count
            for key, value in losses.items()
        }
        target = advantages.flatten() + values.flatten()
        variance = target.var()
        trainer.losses["explained_variance"] = (
            torch.nan
            if variance == 0
            else (1.0 - (target - values.flatten()).var() / variance).item()
        )
        trainer.epoch += 1


def _clone_state(state):
    return tuple(value.detach().clone() for value in state)

def _mask_state(state, terminals: torch.Tensor):
    alive = (1.0 - terminals).reshape(1, -1, 1)
    return tuple(value * alive for value in state)

def _slice_state(state, indices: torch.Tensor):
    return tuple(value[:, indices].detach() for value in state)


def _anneal_learning_rate(trainer) -> None:
    config = trainer.config
    if not config["anneal_lr"] or trainer.epoch <= 0:
        return
    ratio = trainer.epoch / trainer.total_epochs
    minimum = config["learning_rate"] * config["min_lr_ratio"]
    learning_rate = minimum + 0.5 * (
        config["learning_rate"] - minimum
    ) * (1.0 + np.cos(np.pi * ratio))
    trainer.optimizer.param_groups[0]["lr"] = learning_rate


def _record_losses(
    losses,
    policy_loss,
    value_loss,
    entropy,
    old_kl,
    kl,
    clip_fraction,
    ratio,
) -> None:
    losses["policy_loss"] += policy_loss
    losses["value_loss"] += value_loss
    losses["entropy"] += entropy
    losses["old_approx_kl"] += old_kl
    losses["approx_kl"] += kl
    losses["clipfrac"] += clip_fraction
    losses["importance"] += ratio.mean()
