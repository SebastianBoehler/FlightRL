from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def path_provenance(path: Path) -> dict[str, Any]:
    exists = path.exists()
    data: dict[str, Any] = {"path": str(path), "exists": exists}
    if exists and path.is_file():
        stat = path.stat()
        data["size_bytes"] = stat.st_size
        data["sha256"] = sha256_file(path)
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_path_provenance(expected: dict[str, Any]) -> dict[str, Any]:
    path_text = expected.get("path")
    current = path_provenance(Path(path_text)) if path_text else {"path": path_text, "exists": False}
    failure = path_provenance_failure(expected, current)
    return {
        "path": path_text,
        "expected": expected,
        "current": current,
        "passed": failure is None,
        "failure": failure,
    }


def path_provenance_failure(expected: dict[str, Any], current: dict[str, Any]) -> str | None:
    path = expected.get("path")
    if bool(expected.get("exists", False)) != bool(current.get("exists", False)):
        return f"{path}:existence_changed"
    if not expected.get("exists", False):
        return None
    if expected.get("size_bytes") is not None and expected.get("size_bytes") != current.get("size_bytes"):
        return f"{path}:size_changed"
    if expected.get("sha256") and expected.get("sha256") != current.get("sha256"):
        return f"{path}:sha256_changed"
    return None
