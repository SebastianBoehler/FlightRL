from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .config import FlightConfig, load_config

if TYPE_CHECKING:
    from .env import DronePlanarEnv


def make_env(
    config: FlightConfig | str | Path,
    num_envs: int | None = None,
    seed: int = 0,
    emit_logs: bool = True,
    render_mode: str | None = None,
) -> "DronePlanarEnv":
    from .env import DronePlanarEnv

    cfg = load_config(config) if isinstance(config, (str, Path)) else config
    return DronePlanarEnv(cfg, num_envs=num_envs, seed=seed, emit_logs=emit_logs, render_mode=render_mode)
