from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable, Sequence


def build_door_run_provenance(
    *,
    command: Sequence[str],
    started_at_utc: str,
    elapsed_wall_s: float,
    source_report: Path,
    flightrl_root: Path,
    flightrl_source_sha256: dict[str, str],
    puffer_root: Path,
    generated_files: Iterable[Path],
    native_build_fingerprint: dict,
) -> dict:
    return {
        "command": list(command),
        "started_at_utc": started_at_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_wall_s": elapsed_wall_s,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "source_report": str(source_report),
        "source_report_sha256": _file_sha256(source_report),
        "flightrl_git_head": _git_head(flightrl_root),
        "flightrl_source_sha256": dict(flightrl_source_sha256),
        "puffer_root": str(puffer_root),
        "puffer_git_head": _git_head(puffer_root),
        "puffer_tracked_diff_sha256": _git_diff_sha256(puffer_root),
        "generated_puffer_sha256": _manifest(puffer_root, generated_files),
        "native_build_fingerprint": dict(native_build_fingerprint),
    }


def build_file_manifest(
    root: Path,
    paths: Iterable[Path],
) -> dict[str, str]:
    root = root.resolve()
    return {
        str(path.resolve().relative_to(root)): _file_sha256(path)
        for path in sorted({Path(item) for item in paths})
    }


_manifest = build_file_manifest


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_diff_sha256(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=root,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()
