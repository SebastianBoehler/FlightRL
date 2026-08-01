from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


FRAME_INTEGRITY_STATUSES = frozenset(
    {
        "frame_safe",
        "known_corrupt",
        "unreviewed",
    }
)


class FrameIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FrameIntegrityRecord:
    path: Path
    status: str
    evidence: str


class FrameIntegrityRegistry:
    def __init__(self, records: tuple[FrameIntegrityRecord, ...]) -> None:
        self._records = {record.path: record for record in records}
        if len(self._records) != len(records):
            raise ValueError("frame integrity registry contains duplicate paths")

    @property
    def records(self) -> tuple[FrameIntegrityRecord, ...]:
        return tuple(self._records.values())

    def lookup(self, path: str | Path) -> FrameIntegrityRecord | None:
        return self._records.get(Path(path).resolve())

    def require_frame_safe(self, path: str | Path) -> FrameIntegrityRecord:
        resolved = Path(path).resolve()
        record = self._records.get(resolved)
        if record is None:
            raise FrameIntegrityError(
                f"camera dataset is not registered for frame integrity: {resolved}"
            )
        if record.status != "frame_safe":
            raise FrameIntegrityError(
                f"camera dataset has integrity status {record.status!r}: {resolved}; "
                f"evidence: {record.evidence}"
            )
        return record


def load_frame_integrity_registry(
    path: str | Path,
    *,
    root: str | Path | None = None,
) -> FrameIntegrityRegistry:
    registry_path = Path(path)
    payload = json.loads(registry_path.read_text())
    if payload.get("version") != 1:
        raise ValueError("frame integrity registry version must be 1")
    base = Path(root) if root is not None else registry_path.parent
    records = tuple(_parse_record(item, base.resolve()) for item in payload["datasets"])
    return FrameIntegrityRegistry(records)


def _parse_record(payload: object, root: Path) -> FrameIntegrityRecord:
    if not isinstance(payload, dict):
        raise ValueError("frame integrity dataset entries must be objects")
    status = str(payload.get("status", ""))
    if status not in FRAME_INTEGRITY_STATUSES:
        raise ValueError(
            f"frame integrity status must be one of "
            f"{sorted(FRAME_INTEGRITY_STATUSES)}, got {status!r}"
        )
    relative_path = Path(str(payload.get("path", "")))
    if not str(relative_path) or relative_path.is_absolute():
        raise ValueError("frame integrity paths must be non-empty and relative")
    evidence = str(payload.get("evidence", "")).strip()
    if not evidence:
        raise ValueError("frame integrity records require evidence")
    return FrameIntegrityRecord(
        path=(root / relative_path).resolve(),
        status=status,
        evidence=evidence,
    )
