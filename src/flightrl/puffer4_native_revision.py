from __future__ import annotations

from pathlib import Path
import re
import subprocess


def require_clean_puffer_revision(puffer_root: str | Path) -> dict[str, str]:
    root = Path(puffer_root).expanduser().resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("PufferLib dependency revision is unavailable") from exc
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("PufferLib dependency commit is invalid")
    if dirty:
        raise RuntimeError("PufferLib dependency has tracked source changes")
    return {"git_commit": commit}
