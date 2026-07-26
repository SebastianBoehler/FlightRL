from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class VisionActionScale:
    velocity_m_s: float = 0.15
    yawrate_deg_s: float = 60.0

    def normalize(self, vx: np.ndarray, vy: np.ndarray, yawrate: np.ndarray) -> np.ndarray:
        actions = np.column_stack(
            (
                vx / self.velocity_m_s,
                vy / self.velocity_m_s,
                yawrate / self.yawrate_deg_s,
            )
        )
        return np.clip(actions, -1.0, 1.0).astype(np.float32)

    def physical(self, actions: np.ndarray) -> np.ndarray:
        return np.asarray(actions, dtype=np.float32) * np.asarray(
            [self.velocity_m_s, self.velocity_m_s, self.yawrate_deg_s],
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class VisionActionDataset:
    observations: np.ndarray
    actions: np.ndarray
    phases: np.ndarray
    run_ids: np.ndarray
    host_time_s: np.ndarray
    alignment_error_s: np.ndarray
    contract_json: str


def load_aligned_vision_actions(
    vision_paths: Sequence[str | Path],
    telemetry_paths: Sequence[str | Path],
    *,
    scale: VisionActionScale = VisionActionScale(),
    max_alignment_s: float = 0.05,
) -> VisionActionDataset:
    if not vision_paths or len(vision_paths) != len(telemetry_paths):
        raise ValueError("vision_paths and telemetry_paths must be non-empty and have equal length")

    batches = [
        _load_aligned_run(Path(vision), Path(telemetry), run_id, scale, max_alignment_s)
        for run_id, (vision, telemetry) in enumerate(zip(vision_paths, telemetry_paths, strict=True))
    ]
    contracts = {batch.contract_json for batch in batches}
    if len(contracts) != 1:
        raise ValueError("all vision runs must use the same observation contract")
    return VisionActionDataset(
        observations=np.concatenate([batch.observations for batch in batches]),
        actions=np.concatenate([batch.actions for batch in batches]),
        phases=np.concatenate([batch.phases for batch in batches]),
        run_ids=np.concatenate([batch.run_ids for batch in batches]),
        host_time_s=np.concatenate([batch.host_time_s for batch in batches]),
        alignment_error_s=np.concatenate([batch.alignment_error_s for batch in batches]),
        contract_json=contracts.pop(),
    )


def phase_holdout_split(
    dataset: VisionActionDataset,
    *,
    validation_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    validation: list[int] = []
    for run_id in np.unique(dataset.run_ids):
        for phase in np.unique(dataset.phases[dataset.run_ids == run_id]):
            indices = np.flatnonzero((dataset.run_ids == run_id) & (dataset.phases == phase))
            count = max(1, int(round(len(indices) * validation_fraction)))
            validation.extend(indices[-count:].tolist())
    validation_indices = np.asarray(sorted(validation), dtype=np.int64)
    training_mask = np.ones(len(dataset.actions), dtype=bool)
    training_mask[validation_indices] = False
    return np.flatnonzero(training_mask), validation_indices


def _load_aligned_run(
    vision_path: Path,
    telemetry_path: Path,
    run_id: int,
    scale: VisionActionScale,
    max_alignment_s: float,
) -> VisionActionDataset:
    with np.load(vision_path, allow_pickle=False) as vision:
        observations = np.asarray(vision["observations"], dtype=np.float32)
        vision_times = np.asarray(vision["host_time_s"], dtype=np.float64)
        contract_json = str(vision["contract_json"])
    rows = list(csv.DictReader(telemetry_path.open(newline="")))
    if not rows:
        raise ValueError(f"telemetry log is empty: {telemetry_path}")

    telemetry_times = _column(rows, "host_time_s")
    nearest = _nearest_indices(telemetry_times, vision_times)
    errors = np.abs(telemetry_times[nearest] - vision_times)
    flying = _column(rows, "sys.isFlying")[nearest] > 0.5
    upright = _column(rows, "sys.isTumbled")[nearest] < 0.5
    baseline = np.asarray([row["baseline_controls_drone"].lower() == "true" for row in rows])[nearest]
    valid = (errors <= max_alignment_s) & flying & upright & baseline
    if not np.any(valid):
        raise ValueError(f"no aligned in-flight samples in {vision_path}")

    aligned = nearest[valid]
    actions = scale.normalize(
        _column(rows, "baseline_vx_m_s")[aligned],
        _column(rows, "baseline_vy_m_s")[aligned],
        _column(rows, "baseline_yawrate_deg_s")[aligned],
    )
    phases = np.asarray([rows[index]["phase"] for index in aligned])
    return VisionActionDataset(
        observations=observations[valid],
        actions=actions,
        phases=phases,
        run_ids=np.full(len(aligned), run_id, dtype=np.int16),
        host_time_s=vision_times[valid],
        alignment_error_s=errors[valid],
        contract_json=contract_json,
    )


def _column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def _nearest_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(reference, query, side="left").clip(0, len(reference) - 1)
    left = (right - 1).clip(0, len(reference) - 1)
    choose_left = np.abs(reference[left] - query) <= np.abs(reference[right] - query)
    return np.where(choose_left, left, right)
