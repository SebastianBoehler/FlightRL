from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from flightrl.puffer4_edge_schema import (
    EDGE_FRAME_PIXELS,
    EDGE_TELEMETRY_BOUNDS,
    EDGE_TELEMETRY_DIM,
)

from .contract import COVERAGE_OBSERVATION_DIM
from .teacher import ScanAdvanceTeacher


COVERAGE_SEQUENCE_SCHEMA = "flightrl.coverage.sequence.v1"
EVENT_ADVANCE = 0
EVENT_ENTER_SCAN = 1
EVENT_CONTINUE_SCAN = 2
EVENT_RESUME_ADVANCE = 3
EVENT_LABELS = (
    "advance",
    "enter_scan",
    "continue_scan",
    "resume_advance",
)
COVERAGE_SEQUENCE_ARRAYS = (
    "packed_frames",
    "telemetry",
    "teacher_actions",
    "resets",
    "dones",
    "front_clearance_m",
    "event_labels",
    "scene_ids",
    "pair_ids",
)
_PACKED_FRAME_BYTES = EDGE_FRAME_PIXELS // 2
_SPLITS = ("train", "selection", "final")


@dataclass(frozen=True, slots=True)
class CoverageSequenceDataset:
    packed_frames: np.ndarray
    telemetry: np.ndarray
    teacher_actions: np.ndarray
    resets: np.ndarray
    dones: np.ndarray
    front_clearance_m: np.ndarray
    event_labels: np.ndarray
    scene_ids: np.ndarray
    pair_ids: np.ndarray
    metadata: dict[str, object]

    @property
    def shape(self) -> tuple[int, int]:
        return self.resets.shape

    def model_observation(self, step: int) -> torch.Tensor:
        if type(step) is not int or not 0 <= step < self.shape[0]:
            raise ValueError("coverage sequence step is outside the dataset")
        packed = torch.from_numpy(self.packed_frames[step])
        pixels = torch.empty(self.shape[1], EDGE_FRAME_PIXELS, dtype=torch.float32)
        pixels[:, 0::2] = (packed >> 4).to(torch.float32) / 15.0
        pixels[:, 1::2] = (packed & 0x0F).to(torch.float32) / 15.0
        result = torch.cat((pixels, torch.from_numpy(self.telemetry[step])), dim=1)
        if result.shape != (self.shape[1], COVERAGE_OBSERVATION_DIM):
            raise RuntimeError("coverage sequence produced an incompatible observation")
        return result


def coverage_sequence_metadata(
    *, split: str, steps: int, scene_ids: tuple[int, ...]
) -> dict[str, object]:
    if split not in _SPLITS:
        raise ValueError("coverage sequence split is unsupported")
    if type(steps) is not int or steps <= 0:
        raise ValueError("coverage sequence steps must be positive")
    if not scene_ids or any(type(value) is not int or value < 0 for value in scene_ids):
        raise ValueError("coverage sequence scene IDs must be non-negative integers")
    return {
        "schema": COVERAGE_SEQUENCE_SCHEMA,
        "split": split,
        "steps": steps,
        "agents": len(scene_ids),
        "scene_ids": list(scene_ids),
        "observation_contract": "aideck-coverage-policy-v1",
        "teacher": "privileged_front_clearance_scan_advance",
        "authority": "simulation_only",
        "flight_authority": False,
    }


def require_coverage_sequence_dataset(dataset: CoverageSequenceDataset) -> None:
    if type(dataset) is not CoverageSequenceDataset:
        raise TypeError("coverage sequence dataset has an incompatible type")
    if not isinstance(dataset.metadata, dict):
        raise ValueError("coverage sequence metadata must be a mapping")
    steps = dataset.metadata.get("steps")
    agents = dataset.metadata.get("agents")
    if type(steps) is not int or type(agents) is not int:
        raise ValueError("coverage sequence metadata dimensions are invalid")
    prefix = (steps, agents)
    _array(dataset.packed_frames, prefix + (_PACKED_FRAME_BYTES,), np.uint8, "frames")
    _array(dataset.telemetry, prefix + (EDGE_TELEMETRY_DIM,), np.float32, "telemetry")
    _array(dataset.teacher_actions, prefix + (2,), np.float32, "actions")
    _array(dataset.resets, prefix, np.uint8, "resets")
    _array(dataset.dones, prefix, np.uint8, "dones")
    _array(dataset.front_clearance_m, prefix, np.float32, "front clearance")
    _array(dataset.event_labels, prefix, np.uint8, "events")
    _array(dataset.scene_ids, (agents,), np.uint32, "scene IDs")
    _array(dataset.pair_ids, prefix, np.int64, "pair IDs")
    expected = coverage_sequence_metadata(
        split=dataset.metadata.get("split"),
        steps=steps,
        scene_ids=tuple(int(value) for value in dataset.scene_ids),
    )
    if dataset.metadata != expected:
        raise ValueError("coverage sequence metadata does not match its arrays")
    _validate_values(dataset)


def write_coverage_sequence(
    path: str | Path, dataset: CoverageSequenceDataset
) -> Path:
    require_coverage_sequence_dataset(dataset)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **{name: getattr(dataset, name) for name in COVERAGE_SEQUENCE_ARRAYS},
        metadata=json.dumps(dataset.metadata, sort_keys=True, allow_nan=False),
    )
    return output


def load_coverage_sequence(path: str | Path) -> CoverageSequenceDataset:
    with np.load(Path(path), allow_pickle=False) as data:
        expected = {*COVERAGE_SEQUENCE_ARRAYS, "metadata"}
        if set(data.files) != expected:
            raise ValueError("coverage sequence arrays are missing or incompatible")
        dataset = CoverageSequenceDataset(
            **{name: data[name] for name in COVERAGE_SEQUENCE_ARRAYS},
            metadata=json.loads(str(data["metadata"])),
        )
    require_coverage_sequence_dataset(dataset)
    return dataset


def require_matched_counterfactual_pairs(
    dataset: CoverageSequenceDataset,
) -> dict[str, int]:
    require_coverage_sequence_dataset(dataset)
    identifiers = sorted(
        int(value) for value in set(dataset.pair_ids.flat) if value >= 0
    )
    if not identifiers:
        raise ValueError("coverage sequence has no matched counterfactual pairs")
    clear_samples = blocked_samples = history_steps = 0
    for identifier in identifiers:
        locations = np.argwhere(dataset.pair_ids == identifier)
        if locations.shape != (2, 2):
            raise ValueError(
                "coverage counterfactual pair must contain exactly two samples"
            )
        first, second = (tuple(location) for location in locations)
        if dataset.scene_ids[first[1]] != dataset.scene_ids[second[1]]:
            raise ValueError(
                "coverage counterfactual source scene does not match"
            )
        if (
            dataset.resets[first] != 1
            or dataset.resets[second] != 1
            or not np.array_equal(dataset.telemetry[first], dataset.telemetry[second])
        ):
            raise ValueError("coverage counterfactual nonvisual history does not match")
        if np.array_equal(dataset.packed_frames[first], dataset.packed_frames[second]):
            raise ValueError("coverage counterfactual frames must differ")
        if np.array_equal(
            dataset.teacher_actions[first], dataset.teacher_actions[second]
        ):
            raise ValueError("coverage counterfactual teacher labels must differ")
        clearance = (
            dataset.front_clearance_m[first],
            dataset.front_clearance_m[second],
        )
        if min(clearance) > ScanAdvanceTeacher.turn_clearance_m or max(
            clearance
        ) < ScanAdvanceTeacher.resume_clearance_m:
            raise ValueError("coverage counterfactual clearance labels are ambiguous")
        clear, blocked = (
            (first, second) if clearance[0] > clearance[1] else (second, first)
        )
        if not np.array_equal(
            dataset.teacher_actions[clear], np.asarray((0.5, 0.0), dtype=np.float32)
        ) or not np.array_equal(
            dataset.teacher_actions[blocked], np.asarray((0.0, 1.0), dtype=np.float32)
        ):
            raise ValueError(
                "coverage counterfactual teacher mode does not match clearance"
            )
        clear_samples += 1
        blocked_samples += 1
        history_steps += 1
    return {
        "pairs": len(identifiers),
        "clear_samples": clear_samples,
        "blocked_samples": blocked_samples,
        "history_steps": history_steps,
    }


def _array(value: np.ndarray, shape: tuple[int, ...], dtype, label: str) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype != dtype
    ):
        raise ValueError(f"coverage sequence {label} shape or dtype is incompatible")


def _validate_values(dataset: CoverageSequenceDataset) -> None:
    if not np.isfinite(dataset.telemetry).all() or not np.isfinite(
        dataset.teacher_actions
    ).all():
        raise ValueError("coverage sequence contains nonfinite actor values")
    bounds = np.asarray(EDGE_TELEMETRY_BOUNDS, dtype=np.float32)
    if np.any((dataset.telemetry < bounds[:, 0]) | (dataset.telemetry > bounds[:, 1])):
        raise ValueError("coverage sequence telemetry is outside normalized bounds")
    for section in (slice(6, 9), slice(13, 15)):
        if not np.allclose(
            np.linalg.norm(dataset.telemetry[..., section], axis=-1),
            1.0,
            atol=1.0e-4,
            rtol=0.0,
        ):
            raise ValueError("coverage sequence orientation telemetry is invalid")
    if np.any(np.abs(dataset.teacher_actions) > 1.0):
        raise ValueError("coverage sequence actions are outside normalized bounds")
    if np.any((dataset.resets > 1) | (dataset.dones > 1)) or not np.all(
        dataset.resets[0] == 1
    ):
        raise ValueError("coverage sequence reset/done flags are noncanonical")
    if dataset.shape[0] > 1 and not np.array_equal(
        dataset.resets[1:], dataset.dones[:-1]
    ):
        raise ValueError("coverage sequence terminal chronology is invalid")
    if not np.isfinite(dataset.front_clearance_m).all() or np.any(
        dataset.front_clearance_m <= 0.0
    ):
        raise ValueError("coverage sequence front clearance must be positive")
    if np.any(dataset.event_labels >= len(EVENT_LABELS)) or np.any(
        dataset.pair_ids < -1
    ):
        raise ValueError("coverage sequence event or pair labels are invalid")
    forward = np.isin(
        dataset.event_labels,
        np.asarray((EVENT_ADVANCE, EVENT_RESUME_ADVANCE), dtype=np.uint8),
    )
    expected_actions = np.empty_like(dataset.teacher_actions)
    expected_actions[forward] = (0.5, 0.0)
    expected_actions[~forward] = (0.0, 1.0)
    if not np.array_equal(dataset.teacher_actions, expected_actions):
        raise ValueError("coverage sequence teacher action does not match event")
