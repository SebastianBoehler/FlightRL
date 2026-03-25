from __future__ import annotations

from pathlib import Path

import torch
from pufferlib.pufferl import NoLogger, PuffeRL

from .config import FlightConfig
from .env import DronePlanarEnv
from .policy import FlightPolicy


def build_train_config(config: FlightConfig, total_agents: int) -> dict[str, object]:
    train = config.training
    batch_size = max(train.batch_size, total_agents * train.bptt_horizon)
    minibatch_size = min(train.minibatch_size, batch_size)
    max_minibatch_size = max(train.max_minibatch_size, minibatch_size)
    total_timesteps = max(train.total_timesteps, batch_size * train.iterations)
    return {
        "env": f"flight_{config.task.task_type}",
        "name": "flightrl",
        "project": "flightrl",
        "seed": 42,
        "torch_deterministic": train.torch_deterministic,
        "cpu_offload": train.cpu_offload,
        "device": train.device,
        "optimizer": train.optimizer,
        "precision": train.precision,
        "total_timesteps": total_timesteps,
        "batch_size": batch_size,
        "bptt_horizon": train.bptt_horizon,
        "minibatch_size": minibatch_size,
        "max_minibatch_size": max_minibatch_size,
        "learning_rate": train.learning_rate,
        "anneal_lr": train.anneal_lr,
        "gamma": train.gamma,
        "gae_lambda": train.gae_lambda,
        "update_epochs": train.update_epochs,
        "clip_coef": train.clip_coef,
        "vf_coef": train.vf_coef,
        "vf_clip_coef": train.vf_clip_coef,
        "max_grad_norm": train.max_grad_norm,
        "ent_coef": train.ent_coef,
        "adam_beta1": train.adam_beta1,
        "adam_beta2": train.adam_beta2,
        "adam_eps": train.adam_eps,
        "compile": train.compile,
        "compile_mode": train.compile_mode,
        "compile_fullgraph": train.compile_fullgraph,
        "prio_alpha": train.prio_alpha,
        "prio_beta0": train.prio_beta0,
        "vtrace_rho_clip": train.vtrace_rho_clip,
        "vtrace_c_clip": train.vtrace_c_clip,
        "data_dir": train.data_dir,
        "checkpoint_interval": train.checkpoint_interval,
        "use_rnn": False,
        "tag": None,
    }


def create_env_and_policy(
    config: FlightConfig,
    seed: int = 0,
    render_mode: str | None = None,
) -> tuple[DronePlanarEnv, FlightPolicy]:
    env = DronePlanarEnv(config, seed=seed, emit_logs=False, render_mode=render_mode)
    policy = FlightPolicy(env, hidden_size=config.training.hidden_size)
    if config.training.device == "cuda" and torch.cuda.is_available():
        policy = policy.to("cuda")
    return env, policy


def run_training(config: FlightConfig, seed: int = 0) -> list[dict[str, float]]:
    env, policy = create_env_and_policy(config, seed=seed)
    train_config = build_train_config(config, env.num_agents)
    trainer = PuffeRL(train_config, env, policy, logger=NoLogger(train_config))
    logs: list[dict[str, float]] = []
    for iteration in range(config.training.iterations):
        trainer.evaluate()
        trainer.train()
        mean_logs = trainer.mean_and_log()
        if mean_logs:
            logs.append(mean_logs)
        if (iteration + 1) % config.training.checkpoint_interval == 0:
            trainer.save_checkpoint()
    trainer.close()
    return logs


def load_policy_checkpoint(policy: FlightPolicy, checkpoint_path: str | Path, device: str = "cpu") -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
