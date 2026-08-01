from __future__ import annotations

import json

import numpy as np

from flightrl.puffer4_edge_sequence import (
    EdgeSequenceDataset,
    require_edge_sequence_structure,
)


CRITICAL_ACTION_SWITCH = 0.05
EDGE_COVERAGE_MINIMA = {
    "train": {
        "segments": 512,
        "critical_events": 2048,
        "initial_visible": 128,
        "initial_outside_fov": 128,
        **{f"layout_family_{index}": 64 for index in range(4)},
        **{f"door_face_{index}": 64 for index in range(4)},
        "low_light": 32,
        "normal_light": 128,
        "obstacle": 128,
        "obstacle_absent": 128,
    },
    "selection": {
        "segments": 256,
        "initial_visible": 64,
        "initial_outside_fov": 64,
        **{f"layout_family_{index}": 32 for index in range(4)},
        **{f"door_face_{index}": 32 for index in range(4)},
        "low_light": 16,
        "normal_light": 64,
        "obstacle": 64,
        "obstacle_absent": 64,
    },
}


def edge_realized_coverage(dataset: EdgeSequenceDataset) -> dict[str, int]:
    require_edge_sequence_structure(dataset)
    reset = dataset.resets.astype(bool)
    groups = dataset.scene_group_ids[reset]
    outside = (groups & 64) != 0
    low_light = (groups & 16) != 0
    obstacle = (groups & 32) != 0
    report = {
        "segments": int(reset.sum()),
        "critical_events": int(_critical_events(dataset).sum()),
        "initial_visible": int((~outside).sum()),
        "initial_outside_fov": int(outside.sum()),
        "low_light": int(low_light.sum()),
        "normal_light": int((~low_light).sum()),
        "obstacle": int(obstacle.sum()),
        "obstacle_absent": int((~obstacle).sum()),
    }
    layout = groups & 3
    door_face = (groups >> 2) & 3
    for index in range(4):
        report[f"layout_family_{index}"] = int((layout == index).sum())
        report[f"door_face_{index}"] = int((door_face == index).sum())
    return report


def require_edge_training_coverage(
    train: EdgeSequenceDataset,
    selection: EdgeSequenceDataset,
) -> dict[str, dict[str, int]]:
    datasets = {"train": train, "selection": selection}
    report = {}
    deficient = {}
    for split, dataset in datasets.items():
        if dataset.metadata.get("split") != split:
            raise ValueError("edge coverage datasets use incompatible splits")
        realized = edge_realized_coverage(dataset)
        report[split] = realized
        missing = {
            field: {"actual": realized[field], "minimum": minimum}
            for field, minimum in EDGE_COVERAGE_MINIMA[split].items()
            if realized[field] < minimum
        }
        if missing:
            deficient[split] = missing
    if deficient:
        payload = json.dumps(deficient, sort_keys=True, separators=(",", ":"))
        raise ValueError(f"edge realized coverage deficient: {payload}")
    return report


def _critical_events(dataset: EdgeSequenceDataset) -> np.ndarray:
    reset = dataset.resets.astype(bool)
    visible = dataset.grounding[..., 0] > 0.5
    critical = reset.copy()
    if dataset.shape[0] <= 1:
        return critical
    continuation = ~reset[1:]
    critical[1:] |= continuation & (visible[1:] != visible[:-1])
    action_delta = np.max(
        np.abs(dataset.teacher_actions[1:] - dataset.teacher_actions[:-1]),
        axis=-1,
    )
    critical[1:] |= continuation & (action_delta >= CRITICAL_ACTION_SWITCH)
    return critical
