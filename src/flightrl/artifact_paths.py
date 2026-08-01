from __future__ import annotations

from itertools import combinations
from pathlib import Path


def require_distinct_artifact_paths(
    **paths: str | Path,
) -> dict[str, Path]:
    """Resolve artifact paths and reject lexical, symlink, or hardlink aliases."""
    resolved = {
        label: Path(path).expanduser().resolve()
        for label, path in paths.items()
    }
    for (first_label, first), (second_label, second) in combinations(
        resolved.items(),
        2,
    ):
        aliases = first == second or (
            first.exists() and second.exists() and first.samefile(second)
        )
        if aliases:
            raise ValueError(
                f"{first_label} and {second_label} artifact paths must be distinct"
            )
    return resolved
