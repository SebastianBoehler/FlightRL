from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_hardware_blockers(path: Path | None, extra: list[str] | None = None) -> list[str]:
    blockers = list(extra or [])
    if path and path.exists():
        blockers.extend(blockers_from_report(json.loads(path.read_text())))
    return sorted(dict.fromkeys(blocker for blocker in blockers if blocker))


def blockers_from_report(report: dict[str, Any]) -> list[str]:
    raw = report.get("blockers", [])
    if not isinstance(raw, list):
        raise ValueError("hardware blocker report must contain a list field named 'blockers'")
    return [str(item) for item in raw]
