from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


DEFAULT_WANDB_ENV = Path(".secrets/wandb.env")


def add_wandb_args(parser: argparse.ArgumentParser, *, default_project: str = "FlightRL", default_enabled: bool = True) -> None:
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=default_enabled, help="Track this training run with Weights & Biases.")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable Weights & Biases for this run.")
    parser.add_argument("--wandb-project", default=default_project)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-env-file", default=str(DEFAULT_WANDB_ENV))
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")


def init_wandb(args, config: dict[str, Any]):
    if not getattr(args, "wandb", False):
        return None
    load_wandb_env(getattr(args, "wandb_env_file", DEFAULT_WANDB_ENV))
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("--wandb requires the optional 'wandb' package to be installed") from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or None,
        tags=_tags(args.wandb_tags),
        mode=args.wandb_mode,
        config=config,
    )


def log_metrics(run, metrics: dict[str, float], *, step: int | None = None) -> None:
    if run is not None:
        run.log(metrics, step=step)


def log_artifacts(run, *, name: str, paths: list[str | Path], artifact_type: str) -> None:
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(name, type=artifact_type)
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            artifact.add_file(str(candidate))
    run.log_artifact(artifact)


def args_config(args, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    config = {}
    for key, value in vars(args).items():
        if key.startswith("wandb"):
            continue
        config[key] = _jsonable(value)
    return {**config, **(extra or {})}


def load_wandb_env(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.removeprefix("export ").split("=", 1)
        os.environ[key.strip()] = value.strip().strip("\"'")


def _tags(raw: str) -> list[str]:
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
