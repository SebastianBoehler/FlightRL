from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Sequence

from .puffer4_config import Puffer4ExportSettings
from .puffer4_runtime import BUILD_MODE_FLAGS, ensure_puffer_build_matches, normalize_puffer_args, puffer_subprocess_env, resolve_pufferlib_root
from .puffer4_sixdof_export import SixDofPufferExportResult, export_sixdof_puffer4_assets


def export_and_build_sixdof(
    *,
    pufferlib_root: str | Path,
    settings: Puffer4ExportSettings,
    build_mode: str,
    no_build: bool,
) -> tuple[Path, SixDofPufferExportResult]:
    root = resolve_pufferlib_root(pufferlib_root)
    result = export_sixdof_puffer4_assets(root, settings=settings)
    if not no_build:
        subprocess.run(
            ["bash", "build.sh", settings.env_name, *BUILD_MODE_FLAGS[build_mode]],
            cwd=root,
            check=True,
            env=puffer_subprocess_env(build_mode, ()),
        )
    return root, result


def run_sixdof_train(
    *,
    pufferlib_root: str | Path,
    settings: Puffer4ExportSettings,
    build_mode: str,
    no_build: bool = False,
    puffer_args: Sequence[str] = (),
    python_executable: str | None = None,
) -> subprocess.CompletedProcess[None]:
    root, _ = export_and_build_sixdof(
        pufferlib_root=pufferlib_root,
        settings=settings,
        build_mode=build_mode,
        no_build=no_build,
    )
    ensure_puffer_build_matches(root, settings.env_name, no_build=no_build, python_executable=python_executable, build_mode=build_mode)
    forwarded = normalize_puffer_args(puffer_args, build_mode)
    command = [
        python_executable or sys.executable,
        "-m",
        "pufferlib.pufferl",
        "train",
        settings.env_name,
        *forwarded,
    ]
    return subprocess.run(command, cwd=root, check=True, env=puffer_subprocess_env(build_mode, puffer_args))
