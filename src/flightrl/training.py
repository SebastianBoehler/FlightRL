from __future__ import annotations

from pathlib import Path

from .config import FlightConfig
from .env import DronePlanarEnv
from .policy import FlightPolicy, create_policy_for_checkpoint


def create_env_and_policy(
    config: FlightConfig,
    seed: int = 0,
    render_mode: str | None = None,
    *,
    policy_hidden_size: int | None = None,
    policy_num_layers: int = 2,
) -> tuple[DronePlanarEnv, FlightPolicy]:
    env = DronePlanarEnv(config, seed=seed, emit_logs=False, render_mode=render_mode)
    policy = create_policy_for_checkpoint(
        env,
        checkpoint_path=None,
        hidden_size=policy_hidden_size or config.training.hidden_size,
        num_layers=policy_num_layers,
        device=config.training.device,
    )
    if not isinstance(policy, FlightPolicy):
        raise TypeError("unexpected default policy type")
    return env, policy


def load_policy_for_evaluation(
    config: FlightConfig,
    checkpoint_path: str | Path,
    seed: int = 0,
    render_mode: str | None = None,
    *,
    policy_hidden_size: int | None = None,
    policy_num_layers: int = 2,
):
    env = DronePlanarEnv(config, seed=seed, emit_logs=False, render_mode=render_mode)
    policy = create_policy_for_checkpoint(
        env,
        checkpoint_path=checkpoint_path,
        hidden_size=policy_hidden_size or config.training.hidden_size,
        num_layers=policy_num_layers,
        device=config.training.device,
    )
    return env, policy
