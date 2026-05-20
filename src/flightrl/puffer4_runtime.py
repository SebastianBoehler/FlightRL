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


def puffer_subprocess_env(build_mode: str, puffer_args: Sequence[str]) -> dict[str, str]:
    env = os.environ.copy()
    forwarded = normalize_puffer_args(puffer_args, build_mode)
    if build_mode == "cpu" or "--slowly" in forwarded:
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if sys.platform == "darwin":
        clang = Path("/opt/homebrew/opt/llvm/bin/clang")
        clangxx = Path("/opt/homebrew/opt/llvm/bin/clang++")
        if clang.exists() and clangxx.exists():
            env.setdefault("CC", str(clang))
            env.setdefault("CXX", str(clangxx))
    return env


def compiled_puffer_env(root: Path, *, python_executable: str | None = None, build_mode: str = "cpu") -> str | None:
    completed = subprocess.run(
        [
            python_executable or sys.executable,
            "-c",
            "from pufferlib import _C; print(getattr(_C, 'env_name', '') or '')",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=puffer_subprocess_env(build_mode, ()),
    )
    if completed.returncode != 0:
        return None
    compiled = completed.stdout.strip()
    return compiled or None


def ensure_puffer_build_matches(
    root: Path,
    expected_env_name: str,
    *,
    no_build: bool,
    python_executable: str | None = None,
    build_mode: str = "cpu",
) -> None:
    if not no_build:
        return
    compiled = compiled_puffer_env(root, python_executable=python_executable, build_mode=build_mode)
    if compiled is not None and compiled != expected_env_name:
        raise RuntimeError(
            f"PufferLib native extension is built for {compiled!r}, not {expected_env_name!r}; "
            "rerun without --no-build or use the compiled --env-name."
        )


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
            env=puffer_subprocess_env(build_mode, ()),
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
    ensure_puffer_build_matches(root, settings.env_name, no_build=no_build, python_executable=python_executable, build_mode=build_mode)
    command = [
        python_executable or sys.executable,
        "-m",
        "pufferlib.pufferl",
        "train",
        settings.env_name,
        *normalize_puffer_args(puffer_args, build_mode),
    ]
    return subprocess.run(command, cwd=root, check=True, env=puffer_subprocess_env(build_mode, puffer_args))
