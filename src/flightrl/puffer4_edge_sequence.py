from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from flightrl.puffer4_edge_collection_evidence import (
    build_edge_dataset_metadata,
    require_edge_collection_metadata,
)
from flightrl.puffer4_edge_contract import (
    EDGE_ACTION_DIM,
    EDGE_FRAME_PIXELS,
    EDGE_MISSION_TOKEN_COUNT,
    EDGE_OBSERVATION_DIM,
    EDGE_TELEMETRY_BOUNDS,
)
from flightrl.puffer4_edge_dagger import require_edge_execution_trace


edge_dataset_metadata = build_edge_dataset_metadata
_PACKED_FRAME_BYTES = EDGE_FRAME_PIXELS // 2
_SPLIT_ORDER = ("train", "selection", "final")


@dataclass(frozen=True, slots=True)
class EdgeSequenceDataset:
    packed_frames: np.ndarray
    telemetry: np.ndarray
    target_ids: np.ndarray
    teacher_actions: np.ndarray
    behavior_actions: np.ndarray
    execution_student_mask: np.ndarray
    grounding: np.ndarray
    resets: np.ndarray
    dones: np.ndarray
    metadata: dict

    @property
    def shape(self) -> tuple[int, int]:
        return self.target_ids.shape

    def model_observation(self, step: int) -> torch.Tensor:
        if type(step) is not int or not 0 <= step < self.shape[0]:
            raise ValueError("edge dataset step is outside the sequence")
        packed = torch.from_numpy(self.packed_frames[step])
        pixels = torch.empty(
            self.shape[1],
            EDGE_FRAME_PIXELS,
            dtype=torch.float32,
        )
        pixels[:, 0::2] = (packed >> 4).to(torch.float32) / 15.0
        pixels[:, 1::2] = (packed & 0x0F).to(torch.float32) / 15.0
        telemetry = torch.from_numpy(self.telemetry[step])
        mission = torch.nn.functional.one_hot(
            torch.from_numpy(self.target_ids[step]).to(torch.int64),
            EDGE_MISSION_TOKEN_COUNT,
        ).to(torch.float32)
        result = torch.cat((pixels, telemetry, mission), dim=1)
        if result.shape != (self.shape[1], EDGE_OBSERVATION_DIM):
            raise RuntimeError("edge dataset produced an incompatible observation")
        return result


def require_edge_sequence_dataset(dataset: EdgeSequenceDataset) -> None:
    require_edge_sequence_structure(dataset)
    require_edge_execution_trace(dataset)


def require_edge_sequence_structure(dataset: EdgeSequenceDataset) -> None:
    if type(dataset) is not EdgeSequenceDataset:
        raise TypeError("edge sequence dataset has an incompatible type")
    _validate_metadata(dataset.metadata)
    steps = dataset.metadata["steps"]
    agents = dataset.metadata["agents"]
    prefix = (steps, agents)
    _array(dataset.packed_frames, prefix + (_PACKED_FRAME_BYTES,), np.uint8, "frames")
    _array(dataset.telemetry, prefix + (19,), np.float32, "telemetry")
    _array(dataset.target_ids, prefix, np.uint8, "target IDs")
    _array(dataset.teacher_actions, prefix + (EDGE_ACTION_DIM,), np.float32, "actions")
    _array(dataset.behavior_actions, prefix + (EDGE_ACTION_DIM,), np.float32, "behavior")
    _array(dataset.execution_student_mask, (agents,), np.uint8, "execution mask")
    _array(dataset.grounding, prefix + (4,), np.float32, "grounding")
    _array(dataset.resets, prefix, np.uint8, "resets")
    _array(dataset.dones, prefix, np.uint8, "dones")
    _validate_values(dataset)


def write_edge_sequence_dataset(
    path: str | Path,
    dataset: EdgeSequenceDataset,
) -> Path:
    require_edge_sequence_dataset(dataset)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        packed_frames=dataset.packed_frames,
        telemetry=dataset.telemetry,
        target_ids=dataset.target_ids,
        teacher_actions=dataset.teacher_actions,
        behavior_actions=dataset.behavior_actions,
        execution_student_mask=dataset.execution_student_mask,
        grounding=dataset.grounding,
        resets=dataset.resets,
        dones=dataset.dones,
        metadata=json.dumps(dataset.metadata, sort_keys=True, allow_nan=False),
    )
    return output


def load_edge_sequence_dataset(
    path: str | Path,
    *,
    verify_execution_trace: bool = True,
) -> EdgeSequenceDataset:
    with np.load(Path(path), allow_pickle=False) as data:
        if set(data.files) != {
            "packed_frames",
            "telemetry",
            "target_ids",
            "teacher_actions",
            "behavior_actions",
            "execution_student_mask",
            "grounding",
            "resets",
            "dones",
            "metadata",
        }:
            raise ValueError("edge dataset arrays are missing or incompatible")
        dataset = EdgeSequenceDataset(
            packed_frames=data["packed_frames"],
            telemetry=data["telemetry"],
            target_ids=data["target_ids"],
            teacher_actions=data["teacher_actions"],
            behavior_actions=data["behavior_actions"],
            execution_student_mask=data["execution_student_mask"],
            grounding=data["grounding"],
            resets=data["resets"],
            dones=data["dones"],
            metadata=json.loads(str(data["metadata"])),
        )
    if verify_execution_trace:
        require_edge_sequence_dataset(dataset)
    else:
        require_edge_sequence_structure(dataset)
    return dataset


def require_disjoint_edge_datasets(*datasets: EdgeSequenceDataset) -> None:
    _require_disjoint_edge_datasets(datasets, require_edge_sequence_dataset)


def require_disjoint_edge_dataset_structures(
    *datasets: EdgeSequenceDataset,
) -> None:
    _require_disjoint_edge_datasets(datasets, require_edge_sequence_structure)


def _require_disjoint_edge_datasets(datasets, validator) -> None:
    if len(datasets) > len(_SPLIT_ORDER):
        raise ValueError("edge dataset split sequence is not canonical or disjoint")
    physical_seeds: set[int] = set()
    appearance_seeds: set[int] = set()
    for index, dataset in enumerate(datasets):
        validator(dataset)
        physical_seed = dataset.metadata["base_seed"]
        appearance_seed = dataset.metadata["appearance_seed"]
        split = dataset.metadata["split"]
        if split != _SPLIT_ORDER[index]:
            raise ValueError("edge dataset split sequence is not canonical or disjoint")
        if physical_seed in physical_seeds or appearance_seed in appearance_seeds:
            raise ValueError("edge datasets are not seed- and split-disjoint")
        physical_seeds.add(physical_seed)
        appearance_seeds.add(appearance_seed)


def require_matching_edge_dataset_environments(
    *datasets: EdgeSequenceDataset,
) -> None:
    if not datasets:
        raise ValueError("at least one edge dataset environment is required")
    expected = _environment_invariant(datasets[0])
    for dataset in datasets:
        require_edge_sequence_structure(dataset)
        if _environment_invariant(dataset) != expected:
            raise ValueError("edge dataset environments do not match")


def _environment_invariant(dataset: EdgeSequenceDataset) -> dict:
    config = {
        name: value
        for name, value in dataset.metadata["environment_config"].items()
        if name not in {"seed", "appearance_seed"}
    }
    return {
        "environment": dataset.metadata["environment"],
        "target_ids": dataset.metadata["target_ids"],
        "policy_contract_sha256": dataset.metadata["policy_contract_sha256"],
        "collection_profile": dataset.metadata["collection_profile"],
        "environment_config": config,
    }


def _validate_metadata(metadata: object) -> None:
    require_edge_collection_metadata(metadata)


def _array(value: np.ndarray, shape: tuple[int, ...], dtype, label: str) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype != dtype
    ):
        raise ValueError(f"edge dataset {label} shape or dtype is incompatible")


def _validate_values(dataset: EdgeSequenceDataset) -> None:
    if not all(
        np.isfinite(value).all()
        for value in (
            dataset.telemetry,
            dataset.teacher_actions,
            dataset.behavior_actions,
            dataset.grounding,
        )
    ):
        raise ValueError("edge dataset contains nonfinite values")
    for index, (low, high) in enumerate(EDGE_TELEMETRY_BOUNDS):
        values = dataset.telemetry[..., index]
        if np.any((values < low) | (values > high)):
            raise ValueError("edge dataset telemetry is outside normalized bounds")
    if not np.allclose(
        np.linalg.norm(dataset.telemetry[..., 6:9], axis=-1), 1.0, atol=1e-4, rtol=0
    ):
        raise ValueError("edge dataset body-up vector is invalid")
    if not np.allclose(
        np.linalg.norm(dataset.telemetry[..., 13:15], axis=-1), 1.0, atol=1e-4, rtol=0
    ):
        raise ValueError("edge dataset relative-yaw pair is invalid")
    if (
        np.any(dataset.target_ids != 0)
        or np.any(np.abs(dataset.teacher_actions) > 1.0)
        or np.any(np.abs(dataset.behavior_actions) > 1.0)
    ):
        raise ValueError("edge dataset target or action is outside the door contract")
    if np.any(dataset.execution_student_mask > 1):
        raise ValueError("edge dataset execution mask is nonbinary")
    if np.any((dataset.resets > 1) | (dataset.dones > 1)) or not np.all(
        dataset.resets[0] == 1
    ):
        raise ValueError("edge dataset reset/done flags are noncanonical")
    if dataset.shape[0] > 1 and not np.array_equal(
        dataset.resets[1:], dataset.dones[:-1]
    ):
        raise ValueError("edge dataset terminal-to-reset chronology is invalid")
    visible = dataset.grounding[..., 0]
    if np.any((visible != 0.0) & (visible != 1.0)):
        raise ValueError("edge dataset visibility labels must be binary")
    if np.any(np.abs(dataset.grounding[..., 1:3]) > 1.0) or np.any(
        (dataset.grounding[..., 3] < 0.0) | (dataset.grounding[..., 3] > 1.0)
    ):
        raise ValueError("edge dataset grounding is outside normalized bounds")
    if np.any(dataset.grounding[visible == 0.0, 1:] != 0.0):
        raise ValueError("edge dataset absent-target box labels must be zero")
    if np.any(dataset.grounding[visible == 1.0, 3] <= 0.0):
        raise ValueError("edge dataset visible-target scale must be positive")
