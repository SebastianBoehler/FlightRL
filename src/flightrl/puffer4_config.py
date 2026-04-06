from __future__ import annotations

import math
from dataclasses import dataclass

from .config import FlightConfig


def _best_divisor(value: int, upper_bound: int) -> int:
    for candidate in range(min(value, upper_bound), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _align_down(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return max(multiple, (value // multiple) * multiple)


def _format_ini_value(value: int | float | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return format(value, ".12g")


@dataclass(slots=True)
class Puffer4ExportSettings:
    env_name: str = "flightrl"
    total_agents: int | None = None
    num_buffers: int | None = None
    num_threads: int | None = None
    policy_hidden_size: int | None = None
    policy_num_layers: int = 2
    train_seed: int = 42


def build_puffer4_sections(
    config: FlightConfig,
    env_values: dict[str, int | float],
    settings: Puffer4ExportSettings,
) -> dict[str, dict[str, int | float | str]]:
    train = config.training
    horizon = max(1, train.bptt_horizon)
    inferred_agents = max(
        config.environment.num_envs,
        math.ceil(train.batch_size / horizon),
    )
    total_agents = settings.total_agents or inferred_agents
    num_buffers = settings.num_buffers or _best_divisor(total_agents, upper_bound=8)
    num_threads = settings.num_threads or num_buffers
    minibatch_size = _align_down(
        min(train.minibatch_size, total_agents * horizon),
        horizon,
    )
    total_timesteps = max(train.total_timesteps, total_agents * horizon * train.iterations)
    policy_hidden_size = settings.policy_hidden_size or train.hidden_size
    env_section = {"seed": settings.train_seed, **env_values}
    return {
        "base": {
            "env_name": settings.env_name,
            "checkpoint_interval": train.checkpoint_interval,
            "seed": settings.train_seed,
        },
        "vec": {
            "total_agents": total_agents,
            "num_buffers": num_buffers,
            "num_threads": num_threads,
        },
        "env": env_section,
        "policy": {
            "hidden_size": policy_hidden_size,
            "num_layers": settings.policy_num_layers,
            "expansion_factor": 1,
        },
        "torch": {
            "network": "MLP",
            "encoder": "DefaultEncoder",
            "decoder": "DefaultDecoder",
        },
        "train": {
            "gpus": 1,
            "seed": settings.train_seed,
            "total_timesteps": total_timesteps,
            "learning_rate": train.learning_rate,
            "anneal_lr": int(train.anneal_lr),
            "min_lr_ratio": 0.0,
            "gamma": train.gamma,
            "gae_lambda": train.gae_lambda,
            "replay_ratio": train.update_epochs,
            "clip_coef": train.clip_coef,
            "vf_coef": train.vf_coef,
            "vf_clip_coef": train.vf_clip_coef,
            "max_grad_norm": train.max_grad_norm,
            "ent_coef": train.ent_coef,
            "beta1": train.adam_beta1,
            "beta2": train.adam_beta2,
            "eps": train.adam_eps,
            "minibatch_size": minibatch_size,
            "horizon": horizon,
            "vtrace_rho_clip": train.vtrace_rho_clip,
            "vtrace_c_clip": train.vtrace_c_clip,
            "prio_alpha": train.prio_alpha,
            "prio_beta0": train.prio_beta0,
        },
    }


def render_puffer4_ini(sections: dict[str, dict[str, int | float | str]]) -> str:
    blocks: list[str] = []
    for name in ("base", "vec", "env", "policy", "torch", "train"):
        values = sections.get(name)
        if not values:
            continue
        lines = [f"[{name}]"]
        for key, value in values.items():
            lines.append(f"{key} = {_format_ini_value(value)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"
