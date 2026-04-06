from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from .config import FlightConfig
from .puffer4_config import Puffer4ExportSettings
from .puffer4_export import Puffer4ExportResult, export_puffer4_assets


BUILD_MODE_FLAGS = {
    "default": [],
    "float": ["--float"],
    "cpu": ["--cpu"],
}


def resolve_pufferlib_root(pufferlib_root: str | Path | None = None) -> Path:
    candidate = pufferlib_root or os.environ.get("PUFFERLIB_ROOT")
    if candidate is None:
        raise ValueError("set --pufferlib-root or PUFFERLIB_ROOT to an upstream PufferLib 4 checkout")
    root = Path(candidate).expanduser().resolve()
    if not (root / "build.sh").exists():
        raise FileNotFoundError(f"not a PufferLib checkout: {root}")
    return root


def normalize_puffer_args(puffer_args: Sequence[str], build_mode: str) -> list[str]:
    forwarded = list(puffer_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if build_mode == "cpu" and "--slowly" not in forwarded:
        forwarded = ["--slowly", *forwarded]
    return forwarded


def export_and_build(
    config: FlightConfig,
    *,
    pufferlib_root: str | Path,
    settings: Puffer4ExportSettings,
    build_mode: str,
    no_build: bool,
) -> tuple[Path, Puffer4ExportResult]:
    root = resolve_pufferlib_root(pufferlib_root)
    result = export_puffer4_assets(config, root, settings=settings)
    if not no_build:
        subprocess.run(
            ["bash", "build.sh", settings.env_name, *BUILD_MODE_FLAGS[build_mode]],
            cwd=root,
            check=True,
        )
    return root, result


def run_train(
    config: FlightConfig,
    *,
    pufferlib_root: str | Path,
    settings: Puffer4ExportSettings,
    build_mode: str,
    no_build: bool = False,
    puffer_args: Sequence[str] = (),
    python_executable: str | None = None,
) -> subprocess.CompletedProcess[None]:
    root, _ = export_and_build(
        config,
        pufferlib_root=pufferlib_root,
        settings=settings,
        build_mode=build_mode,
        no_build=no_build,
    )
    command = [
        python_executable or sys.executable,
        "-m",
        "pufferlib.pufferl",
        "train",
        settings.env_name,
        *normalize_puffer_args(puffer_args, build_mode),
    ]
    return subprocess.run(command, cwd=root, check=True)
