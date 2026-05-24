from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class CleanResult:
    rows: list[dict[str, str]]
    input_count: int
    kept_count: int
    dropped_count: int
    max_observed_step_speed_m_s: float

    @property
    def dropped_fraction(self) -> float:
        return self.dropped_count / max(self.input_count, 1)


def clean_room_rows(rows: Iterable[Mapping[str, str]], *, max_step_speed_m_s: float) -> CleanResult:
    if max_step_speed_m_s <= 0.0:
        raise ValueError("max_step_speed_m_s must be positive")
    source = [dict(row) for row in rows]
    if not source:
        return CleanResult([], 0, 0, 0, 0.0)

    kept = [source[0]]
    max_speed = 0.0
    for row in source[1:]:
        speed = step_speed_m_s(kept[-1], row)
        max_speed = max(max_speed, speed)
        if speed <= max_step_speed_m_s:
            kept.append(row)

    return CleanResult(
        rows=kept,
        input_count=len(source),
        kept_count=len(kept),
        dropped_count=len(source) - len(kept),
        max_observed_step_speed_m_s=max_speed,
    )


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv_rows(path: Path, rows: list[Mapping[str, str]], fieldnames: list[str]) -> None:
    if not fieldnames and rows:
        fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def step_speed_m_s(previous: Mapping[str, str], current: Mapping[str, str]) -> float:
    dt = _float(current, "host_time_s") - _float(previous, "host_time_s")
    if dt <= 1e-6:
        return float("inf")
    delta = _position(current) - _position(previous)
    return float(np.linalg.norm(delta) / dt)


def _position(row: Mapping[str, str]) -> np.ndarray:
    return np.asarray(
        [
            _float(row, "stateEstimate.x"),
            _float(row, "stateEstimate.y"),
            _float(row, "stateEstimate.z"),
        ],
        dtype=np.float32,
    )


def _float(row: Mapping[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
